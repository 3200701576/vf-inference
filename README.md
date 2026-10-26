# RNFL-to-VF inference

Inference code for predicting 52 visual-field values from RNFL thickness maps.
Model weights and data are not included. Actual prediction requires a compatible
checkpoint; reproduction of the reported results has not been verified without it.

## Install

```bash
python -m pip install -r requirements.txt
```

The dependency versions in `requirements.txt` match the tested environment.

## Input and output

Use a `.npz` file, or a directory containing `.npz` files. Each file must contain
a finite numeric `rnflt` array shaped `(H, W)` or `(1, H, W)`, with both dimensions
at least 32. Existing `tds` labels are optional and ignored. Use the same image
orientation and physical units as the checkpoint expects. Preprocessing is
included; input resolution is preserved.

Output is a CSV containing `sample_id` and 52 predictions (`vf_00` to `vf_51`),
in the point order defined in `coordinates.py`, without output rescaling.

## Run

```bash
python inference.py \
  --checkpoint /path/to/best_model.pth \
  --input /path/to/test_samples \
  --output outputs/predictions.csv
```

The model settings are fixed in the code. Missing or incompatible weights cause
an error. Use `--device cpu` to select CPU execution; other options are listed by
`python inference.py --help`.

## Check execution without weights

```bash
python inference.py --smoke-test --device cpu
```

This uses synthetic input and random parameters to check execution and output
shape. It produces no prediction file and does not validate prediction accuracy.
