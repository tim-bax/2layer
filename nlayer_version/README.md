# N-layer e-prop

Configurable number of two-compartment layers: `input → L1 → L2 → … → L_L → readout`.

## Set the number of layers

- **Variable:** `N_LAYERS` in `run_shd.py` (or `--n_layers` on the command line).
- **Layer sizes:** By default each hidden layer has `N_HIDDEN` units. For custom sizes use `--layer_sizes 42,40,32` (comma-separated; length = number of layers).

## Run on SHD

From repo root (so `data` and `2layer_version` are on the path):

```bash
python nlayer_version/run_shd.py --n_layers 3 --n_hidden 40 --epochs 5
```

Examples:

- 1 layer (same idea as 1layer_version): `--n_layers 1 --n_hidden 64`
- 2 layers (like 2layer_version): `--n_layers 2 --n_hidden 40` or `--layer_sizes 42,40`
- 3+ layers: `--n_layers 4 --n_hidden 32`

## Dependencies

Uses `2layer_version/2comp_uniform.py` for `JAXTwoCompartmentalLayer` and `JAXLIFLayer`. Requires JAX and the repo’s `data` package (SHD loader).
