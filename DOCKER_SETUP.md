# EvoMax Docker Setup

This repository now includes a Docker-friendly EvoMax runner at `python evomax_runner.py --config /path/to/config.json`.

The Dockerfile exposes two useful targets:

- The default `final` target is a lightweight image for fixture-based validation and CSV-override workflows. Its build runs the bundled smoke pipeline and output validator.
- The `gpu` target provides Python 3.11, CUDA 12.4, PyTorch 2.6, ESM-2, ESM-IF, and the supervised-score dependencies needed for full inference on Linux/NVIDIA hosts.

## Build The Image

Smoke/default target:

```bash
docker build --platform linux/amd64 -t evomax:smoke .
```

## Run The Smoke Test

The bundled smoke config uses CSV overrides for GPR, ESM-2, and ESM-IF, so it exercises orchestration, normalization, ranking, and output generation without downloading or running the models. This is not a model-inference test.

```bash
mkdir -p smoke-results

docker run --rm \
  --platform linux/amd64 \
  -v "$PWD/smoke-results:/results" \
  evomax:smoke \
  --config /app/tests/fixtures/smoke_config.json
```

Validate the smoke outputs:

```bash
docker run --rm \
  --platform linux/amd64 \
  -v "$PWD/smoke-results:/results" \
  --entrypoint python \
  evomax:smoke \
  /app/tests/validate_smoke.py \
  --results-dir /results \
  --config /app/tests/fixtures/smoke_config.json \
  --expected /app/tests/fixtures/smoke_expectations.json
```

Expected smoke result:

- `all_single_mutants.csv` contains `57` rows.
- `stage1_top5.csv` contains `5` rows.
- `EvoMax_final_top3.csv` contains `3` rows.
- The top final mutation is `A1Y`.

## Run A Real GPU Job

Create a JSON config next to your mounted input files. Relative paths are resolved relative to the config file location, so a config in `/data` can reference a compatible serialized predictor or `my_structure.pdb` directly.

Build the GPU target on a Linux or Linux-compatible NVIDIA host:

```bash
docker build --platform linux/amd64 --target gpu -t evomax:gpu .
```

Minimal GPU config shape:

```json
{
  "wt_sequence": "PASTE_SEQUENCE_HERE",
  "pdb_path": "my_structure.pdb",
  "pdb_chain_id": "A",
  "gpr_model_path": "GPR_BLOSUM.joblib",
  "device_mode": "auto",
  "top_fraction_mid": 0.015,
  "top_k_mid": null,
  "top_k_final": 100,
  "normalization": "robust_median_iqr",
  "esm2_scoring_mode": "p_mut_only",
  "use_gpr_csv": false,
  "use_esm2_csv": false,
  "use_esmiF_csv": false,
  "resume_runs": false,
  "results_dir": "/results"
}
```

The direct `gpr_model_path` pathway expects a serialized predictor compatible with the runner's three-column WT-residue, position and mutant-residue feature interface. Scores from models using other feature representations should be computed outside the runner and supplied with `use_gpr_csv=true`; see the main README for the accepted CSV schema.

The selected PDB or CIF chain must correspond residue-for-residue and position-for-position to `wt_sequence`.

Run it on an NVIDIA host:

```bash
mkdir -p results model-cache

docker run --rm \
  --gpus all \
  -v "$PWD/data:/data" \
  -v "$PWD/results:/results" \
  -v "$PWD/model-cache:/models" \
  evomax:gpu \
  --config /data/evomax_config.json
```

## Notes

- `colab_evomax.py` and `Colab_EvoMax.ipynb` remain as the Colab reference artifacts.
- The Docker path uses `evomax_runner.py`, which removes the Colab-only syntax and supports headless JSON configuration.
- The container writes outputs to `/results` by default.
- The default image contains only the dependencies needed for CSV-override execution; use the `gpu` target for full model inference.
- The first real ESM-2 or ESM-IF run will download model weights unless you mount reusable Hugging Face and PyTorch caches at `/models`.
- Full inference is not exercised by the CPU smoke workflow because ESM-2 and ESM-IF are computationally intensive and require model downloads, a target structure, and user-supplied supervised scores. An NVIDIA GPU is strongly recommended for this path.
