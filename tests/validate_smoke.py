#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_csv_rows(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing expected CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EvoMax smoke-test outputs.")
    parser.add_argument("--results-dir", required=True, help="Directory containing output CSVs.")
    parser.add_argument("--config", required=True, help="Smoke config JSON path.")
    parser.add_argument("--expected", required=True, help="Expected results JSON path.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    config = json.loads(Path(args.config).read_text())
    expected = json.loads(Path(args.expected).read_text())

    all_mutants = load_csv_rows(results_dir / "all_single_mutants.csv")
    gpr_rows = load_csv_rows(results_dir / "gpr_all.csv")
    esm2_rows = load_csv_rows(results_dir / "esm2_all.csv")
    stage1_rows = load_csv_rows(results_dir / f"stage1_top{config['top_k_mid']}.csv")
    esmif_rows = load_csv_rows(results_dir / f"esmiF_top{config['top_k_mid']}.csv")
    final_rows = load_csv_rows(results_dir / f"EvoMax_final_top{config['top_k_final']}.csv")

    if len(all_mutants) != expected["all_single_mutants_rows"]:
        raise SystemExit(
            f"Expected {expected['all_single_mutants_rows']} all-mutant rows, got {len(all_mutants)}."
        )
    if len(gpr_rows) != expected["all_single_mutants_rows"]:
        raise SystemExit(
            f"Expected {expected['all_single_mutants_rows']} GPR rows, got {len(gpr_rows)}."
        )
    if len(esm2_rows) != expected["all_single_mutants_rows"]:
        raise SystemExit(
            f"Expected {expected['all_single_mutants_rows']} ESM-2 rows, got {len(esm2_rows)}."
        )
    if len(stage1_rows) != expected["stage1_rows"]:
        raise SystemExit(
            f"Expected {expected['stage1_rows']} Stage 1 rows, got {len(stage1_rows)}."
        )
    if len(esmif_rows) != expected["stage1_rows"]:
        raise SystemExit(
            f"Expected {expected['stage1_rows']} ESM-IF rows, got {len(esmif_rows)}."
        )
    if len(final_rows) != expected["final_rows"]:
        raise SystemExit(
            f"Expected {expected['final_rows']} final rows, got {len(final_rows)}."
        )

    top_mutation = final_rows[0]["mutation"] if final_rows else None
    if top_mutation != expected["final_top_mutation"]:
        raise SystemExit(
            f"Expected top mutation {expected['final_top_mutation']}, got {top_mutation}."
        )

    print("Smoke validation passed.")
    print(f"Top final mutation: {top_mutation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

