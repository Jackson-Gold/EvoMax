<div align="center">

# EvoMax
### *Adaptive model-guided protein evolution with sparse data optimizes compact eukaryotic genome editors*

[![Hardware](https://img.shields.io/badge/Full%20pipeline-GPU%20recommended-7C3AED?style=flat-square)](#runtime)
[![Scores](https://img.shields.io/badge/Scores-Supervised%20%7C%20ESM--2%20%7C%20ESM--IF-2563EB?style=flat-square)](#model-specifications)
[![Smoke test](https://github.com/Jackson-Gold/EvoMax/actions/workflows/smoke-test.yml/badge.svg)](https://github.com/Jackson-Gold/EvoMax/actions/workflows/smoke-test.yml)
[![License](https://img.shields.io/badge/License-UPenn%20Non--Commercial-blue?style=flat-square)](LICENSE)

**EvoMax** is a data-efficient mutation-ranking framework that integrates **task-specific supervised activity scores**, **ESM-2**, and **ESM-IF** to prioritize **single-site protein variants** from sparse experimental data, sequence, and structure.

</div>

---

## System Requirements

> **Full model inference is intended for Linux and is practical with an NVIDIA GPU and CUDA.** CPU execution is supported by the runner but may be prohibitively slow for ESM-2 and ESM-IF. The fixture-based smoke test uses precomputed scores and runs without a GPU.

| Requirement | Details |
|---|---|
| **OS** | Linux recommended for full inference; any Docker-compatible host for the smoke test |
| **GPU** | CUDA-compatible NVIDIA GPU strongly recommended for full inference; not required for the smoke test |
| **CUDA** | 12.4 in the supported GPU environment |
| **Python** | 3.11 |

The Dockerfile provides a lightweight, fixture-tested smoke target and a separate CUDA target for full model inference. See [`DOCKER_SETUP.md`](DOCKER_SETUP.md) for build and run instructions.

---

## Overview

EvoMax is a two-stage computational pipeline for exhaustive **single-amino-acid substitution** screening. The framework first performs broad, high-throughput prioritization using a **task-specific supervised activity score** and **ESM-2**, and then refines the top candidates using **ESM-IF** conditioned on the supplied protein structure. Final rankings are generated through configurable score normalization and weighted aggregation, with robust median and interquartile-range scaling as the default. For the Fz2 application, comparison of candidate regression models led to selection of a BLOSUM-based GPR as the supervised component. For a new protein-engineering task, users should compare candidate regression models using held-out or cross-validated performance outside this runner and supply the selected model's precomputed scores through the CSV interface. The direct serialized-model pathway is limited to predictors compatible with the runner's three-column WT-residue, position and mutant-residue feature interface. Each scoring stage can also consume a precomputed CSV override for smoke testing or score reuse.

This repository accompanies the manuscript accepted in principle at Nature Biotechnology.

### Project attribution

**[Shijie Wan](https://github.com/sw152)** is the first author. The project was conceived by Shijie Wan and Xue Sherry Gao. EvoMax construction and predictions were carried out by Shijie Wan, [Jackson Gold](https://github.com/Jackson-Gold), [Pranay Vure](https://github.com/pvure), and Casey S. Mogilevsky in the Xue Sherry Gao Laboratory at the University of Pennsylvania.

---

## Supervised Model Benchmarking and ESM-2 Analyses

<p align="center">
  <img src="docs/figures/figure3_panels_latest.png" alt="Manuscript Figure 3 panels a through e, showing data assembly, supervised-model comparison, and descriptive ESM-2 analyses" width="980">
</p>

<p align="center"><sub><b>Manuscript Figure 3a–e.</b> Data assembly and supervised-model comparison (a–c), with separate descriptive ESM-2 analyses of mutation position and representation space (d,e). Panels d and e visualize ESM-2 outputs; they are not additional inputs to the ranking pipeline.</sub></p>

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Overview](#overview)
- [Supervised Model Benchmarking and ESM-2 Analyses](#supervised-model-benchmarking-and-esm-2-analyses)
- [Runtime](#runtime)
- [Inputs](#inputs)
- [Core Configuration](#core-configuration)
- [Pipeline Logic](#pipeline-logic)
- [Outputs](#outputs)
- [Configuration Details](#configuration-details)
- [Model Specifications](#model-specifications)
- [Graphical Abstract](#graphical-abstract)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

---

## Runtime

| Mode | Runtime considerations |
|---|---|
| Fixture smoke test | Uses precomputed CSV scores; no GPU or model download is required |
| Full model inference | An NVIDIA GPU is strongly recommended; runtime varies with sequence length, Stage 1 shortlist size, device, and whether model weights are cached |

A hardware-specific benchmark will be reported only with the protein length, GPU model, candidate count, and cache state specified.

---

## Inputs

For full inference, mount the required model and structure inputs in **`/data`** and provide their paths in the JSON configuration.

| Input | File / Type | Description |
|---|---|---|
| Wild-type sequence | JSON string | Canonical amino-acid sequence supplied as `wt_sequence` |
| Target structure | `.pdb` or `.cif` | Protein structure file whose selected chain corresponds residue-for-residue and position-for-position to `wt_sequence` (example: `/data/v2.pdb`) |
| Supervised activity source | compatible `.joblib` predictor or `.csv` | A serialized predictor compatible with the runner's WT-residue/position/mutant-residue feature interface, or precomputed scores supplied with `use_gpr_csv=true`; models using other feature representations must use the CSV pathway |
| Model cache | directory mount | Optional persistent Hugging Face and PyTorch cache mounted for ESM-2 and ESM-IF weights; this is not a JSON configuration field |

The ESM-2 and ESM-IF models are loaded internally. Alternatively, any scoring stage can use a precomputed CSV by setting `use_gpr_csv`, `use_esm2_csv`, or `use_esmiF_csv` and the corresponding CSV path. The legacy `gpr` option name denotes the supervised activity-score channel; a CSV generated by another regression model can be supplied through the same interface. The bundled smoke test uses all three CSV overrides.

Each override CSV must contain either a `mutation` column using tokens such as `A12V`, or the three columns `wt`, `pos` and `mut`. The preferred score columns are `GPR_score`, `ESM2_score` and `IF_score` for their respective channels; the generic aliases `score`, `pred_fold`, `ll`, `log_likelihood` and `prob` are also accepted.

---

## Core Configuration

Review the following core values before execution. Parameters not explicitly supplied use the runner defaults.

| Parameter | Description | Example / Default |
|---|---|---|
| `wt_sequence` | Full wild-type amino acid sequence | user-specified |
| `pdb_path` | Path to a `.pdb` or `.cif` structure whose selected chain corresponds exactly to `wt_sequence` | `/data/my_structure.pdb` |
| `pdb_chain_id` | Chain identifier to analyze | `"A"` |
| `gpr_model_path` | Path to a compatible serialized predictor using the runner's three-column mutation feature interface | `/data/GPR_BLOSUM.joblib` |
| `use_gpr_csv` | Use precomputed supervised scores instead of a serialized predictor | `false` |
| `gpr_csv_path` | Path to the supervised-score CSV when `use_gpr_csv=true` | user-specified |
| `device_mode` | Device selection for model inference | `"auto"` |
| `esm2_scoring_mode` | ESM-2 mutation-score definition | `"p_mut_only"` |
| `top_fraction_mid` | Fraction of Stage 1 candidates passed to structural refinement | `0.015` (top 1.5%) |
| `top_k_mid` | Optional fixed-size override for backward compatibility | `null` |
| `normalization` | Score-scaling method used throughout the pipeline | `"robust_median_iqr"` |
| `resume_runs` | Reuse existing per-stage CSV outputs in `results_dir` | `true` |

With `esm2_scoring_mode="p_mut_only"`, the ESM-2 score is the softmax probability assigned to the mutant amino acid at the masked position. The alternative `"delta_logp_mut_minus_wt"` mode returns the mutant log probability minus the wild-type log probability and is not the default.

When the sequence, structure, predictor, scoring mode or weights change, use a new results directory or set `resume_runs=false` to prevent reuse of outputs generated under an earlier configuration.

---

## Pipeline Logic

### 1. Exhaustive Mutation Enumeration
All possible single-site substitutions are generated for the supplied wild-type sequence:

$$L \times 19$$

where $L$ denotes sequence length and 19 corresponds to all non-wild-type amino acid substitutions at each residue position.

### 2. Stage 1 — Screening
Every enumerated mutant is scored using:

- **Supervised activity scores** from a compatible serialized predictor or a precomputed score CSV
- **ESM-2**

These scores are combined to generate an initial ranking and to identify candidates that advance to structural refinement.
By default, the top 1.5% of scored candidates are advanced, calculated as
`max(1, floor(N × 0.015))` for `N` scored single-site substitutions. For the
9,405 candidates generated from a 495-residue protein, this yields 141 Stage 1
candidates. A fixed `top_k_mid` can be supplied only when an explicit override
is required.

### 3. Stage 2 — Structural Refinement
The top-ranked Stage 1 candidates are rescored using **ESM-IF**, conditioned on the supplied protein backbone and selected chain.

### 4. Final Ranking
By default, all relevant scores are normalized with:

- `robust_median_iqr`

The normalized scores are then weighted and aggregated into the final mutation ranking.

The bundled runner uses the Round 3 ranking configuration.

---

## Outputs

By default, outputs are written to **`/results`**; this location can be changed with `results_dir`.

| File | Description |
|---|---|
| `all_single_mutants.csv` | Exhaustive list of all enumerated single-site mutations |
| `gpr_all.csv` | Supervised activity scores for all mutants (legacy filename) |
| `esm2_all.csv` | ESM-2 scores generated by the model or supplied as a CSV override |
| `stage1_top{K}.csv` | Top candidates selected after Stage 1 |
| `esmiF_top{K}.csv` | ESM-IF scores for Stage 2 candidates |
| `EvoMax_final_top{K}.csv` | **Final ranked mutation set**, containing up to the configured `top_k_final` candidates |

---

## Configuration Details

### Stage 1 Filtering

| Parameter | Default | Description |
|---|---:|---|
| `top_fraction_mid` | `0.015` | Fraction of candidates advanced to Stage 2 (top 1.5%) |
| `top_k_mid` | `null` | Optional fixed-size override; when set, it takes precedence over `top_fraction_mid` |
| `top_k_final` | `100` | Maximum number of final ranked mutations returned |

### Scoring Weights

| Parameter | Default | Description |
|---|---:|---|
| `w_gpr_s1` | `0.35` | Supervised-score contribution during Stage 1 (legacy parameter name) |
| `w_esm2_s1` | `0.65` | ESM-2 contribution during Stage 1 |
| `w_gpr_final` | `0.05` | Supervised-score contribution in final scoring (legacy parameter name) |
| `w_esm2_final` | `0.70` | ESM-2 contribution in final scoring |
| `w_esmiF_final` | `0.25` | ESM-IF contribution in final scoring |

### Normalization

By default, scores are normalized with:

- `robust_median_iqr` — robust median and interquartile-range scaling

Other supported methods:

- `zscore`
- `rank_percentile`

---

## Model Specifications

| Component | Model | Description |
|---|---|---|
| Supervised activity channel | Task-specific | A compatible serialized predictor or precomputed CSV |
| Evolutionary model | `esm2_t33_650M_UR50D` | 650M-parameter masked language model |
| Structural model | `esm_if1_gvp4_t16_142M_UR50` | 142M-parameter inverse folding model loaded internally |

---

## Graphical Abstract

<p align="center">
  <img src="docs/figures/figure3f_workflow_latest.png" alt="EvoMax computational workflow integrating sequence and structure inputs with GPR, ESM-2, and ESM-IF" width="760">
</p>

<p align="center"><sub><b>Manuscript Figure 3f.</b> Compact graphical summary of the EvoMax workflow integrating sequence input, structure input, GPR, ESM-2, and ESM-IF for iterative high-throughput screening and final mutation ranking.</sub></p>

---

## Citation

If you use EvoMax, please cite the accompanying paper:

```bibtex
@article{evomax2026,
  title   = {Adaptive model-guided protein evolution with sparse data optimizes compact eukaryotic genome editors},
  author  = {Wan, Shijie and Gold, Jackson and Vure, Pranay and Mogilevsky, Casey S. and Talikoti, Ananya and Chen, Tianrong and Gupta, Aman and Biswas, Trisha and You, Zheng and Acharya, Vir and Chatterjee, Pranam and Wang, Xiao and Gao, Xue},
  year    = {2026},
  note    = {Accepted in principle at Nature Biotechnology}
}
```

The archived software record is available through the [Zenodo concept DOI](https://doi.org/10.5281/zenodo.21083837).

---

## License

This software is released under the [University of Pennsylvania Non-Commercial License](LICENSE). For commercial licensing inquiries, contact the [Penn Center for Innovation](https://www.upenn.edu/research/centers/penn-center-for-innovation) at 215-898-9591.

---

## Contact

For questions about EvoMax, please [open an issue](https://github.com/Jackson-Gold/EvoMax/issues) or contact the corresponding author listed in the paper.
