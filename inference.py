#!/usr/bin/env python3
"""Predict 52 VF values from RNFL NPZ samples using an external checkpoint."""
import argparse
import csv
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

from coordinates import VF_COORDS
from model import MultiScaleRNFLToVF
from preprocessing import input_files, load_input, preprocess

def build_model():
    return MultiScaleRNFLToVF(vf_coords=VF_COORDS).eval()


def load_checkpoint(model, checkpoint_path):
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}. Model weights are not distributed.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint
    if isinstance(checkpoint, Mapping):
        for key in ('model_state_dict', 'state_dict'):
            if key in checkpoint:
                state = checkpoint[key]
                break
    if not isinstance(state, Mapping) or not state:
        raise ValueError('Expected a model state_dict or a checkpoint containing one.')
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError('The model state_dict must map parameter names to tensors.')
    if all(key.startswith('module.') for key in state):
        state = {key.removeprefix('module.'): value for key, value in state.items()}
    # The original backbone creates this unused parameter during its first forward.
    # Preserve it when present so loading stays strict; it does not enter prediction.
    if 'backbone.pos_embed' in state:
        value = state['backbone.pos_embed']
        if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < 1 or value.shape[2] != model.d_model:
            raise ValueError('Invalid backbone.pos_embed shape.')
        model.backbone.register_parameter('pos_embed', torch.nn.Parameter(torch.empty_like(value)))
    model.load_state_dict(state, strict=True)


def run_predictions(model, files, device, batch_size, output):
    predictions = []
    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, len(files), batch_size):
            paths = files[start:start + batch_size]
            samples = [load_input(path) for path in paths]
            if len({tuple(sample.shape) for sample in samples}) != 1:
                raise ValueError('A batch contains different image sizes; use --batch-size 1.')
            result = model(torch.stack(samples).to(device))
            if result.shape != (len(paths), 52) or not torch.isfinite(result).all():
                raise RuntimeError('The model returned invalid predictions.')
            predictions.extend(result.cpu().tolist())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental overwrites.
    with output.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id'] + [f'vf_{i:02d}' for i in range(52)])
        for path, values in zip(files, predictions):
            writer.writerow([path.stem] + values)
    return len(predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--input', type=Path, help='One NPZ file or a directory of NPZ files.')
    parser.add_argument('--output', type=Path, default=Path('outputs/predictions.csv'))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--threads', type=int, default=4, help='CPU intra-op threads.')
    parser.add_argument('--smoke-test', action='store_true', help='Synthetic shape check; no valid predictions.')
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('--batch-size and --threads must be positive.')
    if args.smoke_test and (args.checkpoint or args.input):
        parser.error('--smoke-test cannot be combined with --checkpoint or --input.')
    if not args.smoke_test and (args.checkpoint is None or args.input is None):
        parser.error('Prediction requires --checkpoint and --input; weights are not included.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu.')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    torch.set_num_threads(args.threads)
    if not args.smoke_test:
        if not args.checkpoint.is_file():
            parser.error('Checkpoint file does not exist; weights are not included.')
        if args.output.exists():
            parser.error('Output already exists; choose another --output path.')
        files = input_files(args.input)
    else:
        torch.manual_seed(0)
    model = build_model()
    if args.smoke_test:
        rng = np.random.default_rng(0)
        sample = preprocess(rng.uniform(0, 200, (224, 224)).astype(np.float32))
        with torch.inference_mode():
            output = model.to(device)(sample.unsqueeze(0).to(device))
        if output.shape != (1, 52) or not torch.isfinite(output).all():
            raise RuntimeError('Smoke test failed.')
        print('PASS: synthetic input -> (1, 52). Random parameters; no scientific predictions or accuracy validation.')
        return
    load_checkpoint(model, args.checkpoint)
    count = run_predictions(model, files, device, args.batch_size, args.output)
    print(f'Saved {count} predictions to {args.output}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
