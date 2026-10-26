"""Neural network components for inference; parameter names preserve checkpoint compatibility."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.models as models
from torchvision.models.feature_extraction import create_feature_extractor

class CrossAttention(nn.Module):

    def __init__(self, hidden_size: int=256, num_heads: int=8, dropout: float=0.0):
        super().__init__()
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def _shape(self, x, B, seq_len):
        return x.view(B, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, query_tokens, context_tokens, return_attn: bool=False):
        (B, Nq, _) = query_tokens.size()
        (_, Nk, _) = context_tokens.size()
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        self._debug_count += 1
        if self._debug_count <= 3:
            print(f'[CrossAttention] query: {query_tokens.shape}, context: {context_tokens.shape}')
        q = self.query(query_tokens)
        k = self.key(context_tokens)
        v = self.value(context_tokens)
        q = self._shape(q, B, Nq)
        k = self._shape(k, B, Nk)
        v = self._shape(v, B, Nk)
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, Nq, self.all_head_size)
        out = self.out(context)
        out = self.proj_dropout(out)
        if return_attn:
            return (out, attn_probs)
        return out

class MultiLayerCrossAttention(nn.Module):

    def __init__(self, hidden_size: int=256, num_heads: int=8, dropout: float=0.0, num_layers: int=2):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([CrossAttention(hidden_size=hidden_size, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, hidden_size * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_size * 2, hidden_size), nn.Dropout(dropout)) for _ in range(num_layers)])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])

    def forward(self, query_tokens, context_tokens, return_attn: bool=False):
        x = query_tokens
        last_attn_probs = None
        for i in range(self.num_layers):
            if return_attn and i == self.num_layers - 1:
                (attn_out, last_attn_probs) = self.layers[i](x, context_tokens, return_attn=True)
            else:
                attn_out = self.layers[i](x, context_tokens)
            x = self.layer_norms[i](x + attn_out)
            ffn_out = self.ffns[i](x)
            x = self.ffn_norms[i](x + ffn_out)
        if return_attn:
            return (x, last_attn_probs)
        return x

class VFSpatialSelfAttention(nn.Module):

    def __init__(self, hidden_size: int=256, num_heads: int=8, dropout: float=0.0, vf_coords=None, sigma: float=0.3):
        super().__init__()
        assert hidden_size % num_heads == 0, 'hidden_size must be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.all_head_size = hidden_size
        self.sigma = sigma
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(hidden_size)
        if vf_coords is not None:
            if isinstance(vf_coords, torch.Tensor):
                coords = vf_coords.clone().detach().float()
            else:
                coords = torch.tensor(vf_coords, dtype=torch.float32)
            num_vf = coords.size(0)
            coords_expanded_i = coords.unsqueeze(1)
            coords_expanded_j = coords.unsqueeze(0)
            dist_matrix = torch.norm(coords_expanded_i - coords_expanded_j, dim=2)
            distance_bias = -dist_matrix ** 2 / (2 * sigma ** 2)
            self.register_buffer('distance_bias', distance_bias)
        else:
            self.register_buffer('distance_bias', None)

    def _shape(self, x, B, seq_len):
        return x.view(B, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

    def forward(self, vf_tokens):
        (B, num_vf, _) = vf_tokens.size()
        identity = vf_tokens
        q = self.query(vf_tokens)
        k = self.key(vf_tokens)
        v = self.value(vf_tokens)
        q = self._shape(q, B, num_vf)
        k = self._shape(k, B, num_vf)
        v = self._shape(v, B, num_vf)
        attn_scores = torch.matmul(q, k.transpose(-1, -2))
        attn_scores = attn_scores / math.sqrt(self.head_dim)
        if self.distance_bias is not None:
            attn_scores = attn_scores + self.distance_bias.unsqueeze(0).unsqueeze(0)
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.attn_dropout(attn_probs)
        context = torch.matmul(attn_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, num_vf, self.all_head_size)
        out = self.out(context)
        out = self.proj_dropout(out)
        out = self.norm(out + identity)
        return out

class PointwiseConvOutputHead(nn.Module):

    def __init__(self, d_model: int=256, dropout: float=0.3):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(dropout))
        self.conv1d = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1, padding_mode='replicate')
        self.fc_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1)
        out = self.fc_out(x)
        return out.squeeze(-1)

class ParallelFusionBlock(nn.Module):

    def __init__(self, d_model: int, num_scales: int, num_vf: int=52, droppath: float=0.1, alpha_init: float=0.5):
        super().__init__()
        self.num_scales = num_scales
        self.num_vf = num_vf
        self.d_model = d_model
        self.drop_prob = droppath
        self.fusion_channel_mask = None
        self.scale_proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model)) for _ in range(num_scales)])
        self.scale_factors = nn.Parameter(torch.zeros(num_vf, num_scales))
        with torch.no_grad():
            self.scale_factors[:, 0] = 0.5
            self.scale_factors[:, 1] = 0.0
            self.scale_factors[:, 2] = 0.0
        self.fuse_q = nn.Linear(d_model, d_model)
        self.fuse_kv = nn.Linear(d_model, d_model * 2)
        self.fuse_norm = nn.LayerNorm(d_model)
        self.gate = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Sigmoid())
        self.update_norm = nn.LayerNorm(d_model)
        self.alpha_log = nn.Parameter(torch.tensor(math.log(alpha_init / (1 - alpha_init + 1e-06))))
        self.alpha_scale = 0.8
        self.scale_gates = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Sigmoid()) for _ in range(num_scales)])
        self.beta_log = nn.Parameter(torch.tensor(-0.7))

    def set_fusion_channel_mask(self, mask):
        if mask is None:
            self.fusion_channel_mask = None
            return
        mask = [int(c) for c in mask]
        if len(mask) != self.num_scales:
            raise ValueError(f'fusion channel mask length {len(mask)} != num_scales {self.num_scales}')
        if any((c < -1 or c >= self.num_scales for c in mask)):
            raise ValueError(f'fusion channel mask values must be within [-1, {self.num_scales}); -1 means inactive')
        self.fusion_channel_mask = mask

    def forward(self, P_in: torch.Tensor, deltas: list[torch.Tensor]):
        (B, N, D) = P_in.shape
        S = len(deltas)
        if self.fusion_channel_mask is not None:
            active = set(self.fusion_channel_mask)
            deltas = [delta if i in active else torch.zeros_like(delta) for (i, delta) in enumerate(deltas)]
        proj_deltas = []
        for i in range(S):
            proj = self.scale_proj[i](deltas[i])
            proj_deltas.append(proj)
        raw_weights = self.scale_factors.unsqueeze(0).expand(B, -1, -1)
        temperature = 1.0
        w = F.softmax(raw_weights / temperature, dim=-1)
        self.saved_weights = w.detach().cpu()
        stack = torch.stack(proj_deltas, dim=2)
        weighted = stack * w.unsqueeze(-1)
        fused = weighted.sum(dim=2)
        q = self.fuse_q(fused)
        kv = self.fuse_kv(fused)
        (k, v) = kv.chunk(2, dim=-1)
        scale = q.size(-1) ** 0.5
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = F.dropout(attn_probs, p=self.drop_prob * 0.5, training=False)
        attn_out = torch.matmul(attn_probs, v)
        fused = self.fuse_norm(fused + attn_out)
        scale_direct = 0.0
        for i in range(S):
            g_i = self.scale_gates[i](deltas[i])
            scale_direct = scale_direct + g_i * deltas[i]
        beta = torch.sigmoid(self.beta_log)
        fused = fused + beta * scale_direct
        g = self.gate(P_in)
        update = g * fused
        update = self.update_norm(update)
        alpha = torch.sigmoid(self.alpha_log) * self.alpha_scale
        P_out = P_in + alpha * update
        return P_out


class FeatureExtractorBackbone(nn.Module):
    """ResNet-34 with a single-channel stem and three feature outputs."""

    def __init__(self):
        super().__init__()
        network = models.resnet34(weights=None)
        old = network.conv1
        network.conv1 = nn.Conv2d(
            1, old.out_channels, kernel_size=old.kernel_size,
            stride=old.stride, padding=old.padding, bias=False,
        )
        with torch.no_grad():
            network.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))
        self.extractor = create_feature_extractor(
            network, return_nodes={'layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'}
        )

    def forward(self, x):
        features = self.extractor(x)
        return [features['feat2'], features['feat3'], features['feat4']]


class TriBranchBackbone(nn.Module):
    """Three independent single-channel ResNet branches."""

    def __init__(self):
        super().__init__()
        self.branch_thick = FeatureExtractorBackbone()
        self.branch_text = FeatureExtractorBackbone()
        self.branch_curv = FeatureExtractorBackbone()

    def forward(self, x):
        return {
            'thick': self.branch_thick(x[:, 0:1])[2],
            'text': self.branch_text(x[:, 1:2])[2],
            'curv': self.branch_curv(x[:, 2:3])[2],
        }


class MultiScaleRNFLToVF(nn.Module):
    """Selected polar-MLP, pooled-feature, adaptive-fusion inference variant.

    Unused parameterized heads and attention modules remain registered so that
    full checkpoints can be loaded strictly. Only the final prediction is computed.
    """

    def __init__(self, vf_coords):
        super().__init__()
        # Fixed values from the selected experiment configuration.
        d_model, num_heads, num_cross_attn_layers = 384, 8, 2
        dropout, sigma = 0.4, 0.3
        self.d_model = d_model
        self.backbone = TriBranchBackbone()
        branches = ('thick', 'text', 'curv')
        self.branch_proj = nn.ModuleDict({
            name: nn.Conv2d(512, d_model, 1, bias=False) for name in branches
        })

        def attention():
            if num_cross_attn_layers == 1:
                return CrossAttention(hidden_size=d_model, num_heads=num_heads, dropout=dropout)
            return MultiLayerCrossAttention(
                hidden_size=d_model, num_heads=num_heads,
                dropout=dropout, num_layers=num_cross_attn_layers,
            )

        self.branch_attn = nn.ModuleDict({name: attention() for name in branches})
        self.branch_norms = nn.ModuleDict({name: nn.LayerNorm(d_model) for name in branches})
        self.branch_heads = nn.ModuleDict({name: nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Dropout(dropout * 0.3),
            nn.Linear(d_model // 2, 1),
        ) for name in branches})
        self.parallel_fusion = ParallelFusionBlock(
            d_model=d_model, num_scales=3, num_vf=52, droppath=0.1, alpha_init=0.5,
        )
        self.vf_queries = nn.Parameter(torch.randn(1, 52, d_model))
        self.vf_pos_mlp = nn.Sequential(
            nn.Linear(2, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, d_model), nn.GELU(),
        )
        coords = torch.as_tensor(vf_coords, dtype=torch.float32).clone()
        normalized = coords.clone()
        normalized[:, 0] = (coords[:, 0] - 5.0) / 4.0
        normalized[:, 1] = (coords[:, 1] - 4.5) / 3.5
        self.register_buffer('vf_coords', normalized)
        center_x, center_y = torch.tensor(5.0), torch.tensor(4.5)
        dx, dy = coords[:, 0] - center_x, coords[:, 1] - center_y
        r = torch.sqrt(dx ** 2 + dy ** 2)
        r_max = torch.sqrt((9.0 - center_x) ** 2 + (1.0 - center_y) ** 2)
        theta = (torch.atan2(dy, dx) + math.pi) / (2 * math.pi)
        self.register_buffer('vf_polar_coords', torch.stack([r / r_max, theta], dim=1))
        self.scale_attn = nn.ModuleList([attention() for _ in range(3)])
        self.scale_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(3)])
        self.vf_self_attn = VFSpatialSelfAttention(
            hidden_size=d_model, num_heads=num_heads, dropout=dropout,
            vf_coords=self.vf_coords, sigma=sigma,
        )
        self.out_linear = PointwiseConvOutputHead(d_model=d_model, dropout=dropout)

    def forward(self, x):
        if self.training:
            raise RuntimeError('This module supports inference only; call model.eval().')
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError('Expected input shape (batch, 3, height, width).')
        features = self.backbone(x)
        pos = self.vf_pos_mlp(self.vf_polar_coords).unsqueeze(0)
        query = self.vf_queries.expand(x.shape[0], -1, -1) + pos
        tokens = []
        for name in ('thick', 'text', 'curv'):
            feature = self.branch_proj[name](features[name]).flatten(2).permute(0, 2, 1)
            pooled = feature.mean(dim=1, keepdim=True).expand(-1, 52, -1) + pos
            tokens.append(self.vf_self_attn(pooled))
        fused = self.parallel_fusion(query, tokens)
        return self.out_linear(fused)
