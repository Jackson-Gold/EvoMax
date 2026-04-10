#!/usr/bin/env python3
# Copyright (C) 2026 The Trustees of the University of Pennsylvania.
# Licensed under the Penn Software License (non-commercial). See LICENSE file.
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

AA20 = "ACDEFGHIKLMNPQRSTVWY"
SUPPORTED_DEVICE_MODES = {"auto", "cuda", "cpu"}
SUPPORTED_NORMALIZATIONS = {"zscore", "robust_median_iqr", "rank_percentile"}
SUPPORTED_ESMIF_VARIANTS = {"esm_if1_gvp4_t16_142M_UR50"}
PATH_CONFIG_KEYS = {
    "results_dir",
    "pdb_path",
    "gpr_model_path",
    "gpr_csv_path",
    "esm2_csv_path",
    "esmiF_csv_path",
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "wt_sequence": "",
    "pdb_path": "/data/input.pdb",
    "pdb_chain_id": "A",
    "device_mode": "auto",
    "top_k_mid": 100,
    "top_k_final": 100,
    "gpr_model_path": "/data/GPR_BLOSUM.joblib",
    "w_gpr_s1": 0.35,
    "w_esm2_s1": 0.65,
    "w_gpr_final": 0.05,
    "w_esm2_final": 0.70,
    "w_esmiF_final": 0.25,
    "normalization": "robust_median_iqr",
    "use_gpr_csv": False,
    "gpr_csv_path": "",
    "use_esm2_csv": False,
    "esm2_csv_path": "",
    "use_esmiF_csv": False,
    "esmiF_csv_path": "",
    "esm2_size": "esm2_t33_650M_UR50D",
    "esm_if_variant": "esm_if1_gvp4_t16_142M_UR50",
    "esm2_scoring_mode": "p_mut_only",
    "checkpoint_every": 20,
    "resume_runs": True,
    "results_dir": "/results",
}


def resolve_relative_path(base_dir: Path, value: Any) -> str:
    if value in (None, ""):
        return ""
    path = Path(str(value))
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def load_config(config_path: str) -> Dict[str, Any]:
    path = Path(config_path).resolve()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("Config JSON must be an object.")

    unknown = sorted(set(payload) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")

    config = dict(DEFAULT_CONFIG)
    config.update(payload)
    config["wt_sequence"] = str(config["wt_sequence"]).strip().upper()

    for key in PATH_CONFIG_KEYS:
        if key == "results_dir" and not config.get(key):
            config[key] = str((path.parent / "results").resolve())
            continue
        config[key] = resolve_relative_path(path.parent, config.get(key))

    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    if not config["wt_sequence"]:
        raise ValueError("wt_sequence is required.")

    bad = sorted(set(config["wt_sequence"]) - set(AA20))
    if bad:
        raise ValueError(
            f"Found non-canonical residues in wt_sequence: {bad}. "
            "Only the 20 canonical amino acids are supported."
        )

    if config["device_mode"] not in SUPPORTED_DEVICE_MODES:
        raise ValueError(f"device_mode must be one of {sorted(SUPPORTED_DEVICE_MODES)}.")

    if config["normalization"] not in SUPPORTED_NORMALIZATIONS:
        raise ValueError(
            f"normalization must be one of {sorted(SUPPORTED_NORMALIZATIONS)}."
        )

    if config["esm2_scoring_mode"] not in {"p_mut_only", "delta_logp_mut_minus_wt"}:
        raise ValueError(
            "esm2_scoring_mode must be 'p_mut_only' or 'delta_logp_mut_minus_wt'."
        )

    if config["esm_if_variant"] not in SUPPORTED_ESMIF_VARIANTS:
        raise ValueError(
            f"esm_if_variant must be one of {sorted(SUPPORTED_ESMIF_VARIANTS)}."
        )

    for key in ("top_k_mid", "top_k_final", "checkpoint_every"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be a positive integer.")

    if config["use_gpr_csv"] and not config["gpr_csv_path"]:
        raise ValueError("gpr_csv_path is required when use_gpr_csv is true.")
    if not config["use_gpr_csv"] and not config["gpr_model_path"]:
        raise ValueError("gpr_model_path is required when use_gpr_csv is false.")

    if config["use_esm2_csv"] and not config["esm2_csv_path"]:
        raise ValueError("esm2_csv_path is required when use_esm2_csv is true.")

    if config["use_esmiF_csv"] and not config["esmiF_csv_path"]:
        raise ValueError("esmiF_csv_path is required when use_esmiF_csv is true.")
    if not config["use_esmiF_csv"] and not config["pdb_path"]:
        raise ValueError("pdb_path is required when use_esmiF_csv is false.")


def parse_mutation(token: str) -> Tuple[str, int, str]:
    match = re.fullmatch(r"([A-Z])(\d+)([A-Z])", token)
    if not match:
        raise ValueError(f"Bad mutation token: {token}")
    return match.group(1), int(match.group(2)), match.group(3)


def enumerate_single_mutants(sequence: str):
    import pandas as pd

    records = []
    for pos, wt in enumerate(sequence, start=1):
        for aa in AA20:
            if aa == wt:
                continue
            records.append(
                {
                    "mutation": f"{wt}{pos}{aa}",
                    "wt": wt,
                    "pos": pos,
                    "mut": aa,
                }
            )
    frame = pd.DataFrame(records)
    frame["L"] = len(sequence)
    return frame


def apply_mutation(sequence: str, pos: int, mut: str) -> str:
    arr = list(sequence)
    arr[pos - 1] = mut
    return "".join(arr)


def eta_str(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def save_csv(df, path: str, to_print: bool = True) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(destination, index=False)
    if to_print:
        print(f"Saved: {destination}")


def try_read_csv(path: str, required: Iterable[str] | None = None):
    import pandas as pd

    if not path or not os.path.exists(path):
        return None
    frame = pd.read_csv(path)
    if required and not set(required).issubset(frame.columns):
        missing = sorted(set(required) - set(frame.columns))
        raise ValueError(f"CSV {path} is missing required columns: {missing}")
    return frame


def standardize_mutation_columns(df, score_col: str, rename_to: str):
    if df is None:
        raise ValueError("Expected a DataFrame, got None.")

    frame = df.copy()
    if "mutation" not in frame.columns:
        if {"wt", "pos", "mut"}.issubset(frame.columns):
            frame["mutation"] = (
                frame["wt"].astype(str)
                + frame["pos"].astype(int).astype(str)
                + frame["mut"].astype(str)
            )
        else:
            raise ValueError(
                f"{rename_to} CSV must contain 'mutation' or the columns wt, pos, mut."
            )

    resolved_score_col = score_col
    if resolved_score_col not in frame.columns:
        for alt in ("score", "pred_fold", "ll", "log_likelihood", "prob"):
            if alt in frame.columns:
                resolved_score_col = alt
                break
        else:
            raise ValueError(
                f"Could not find score column '{score_col}' in CSV columns {list(frame.columns)}."
            )

    frame = frame[["mutation", resolved_score_col]].dropna().drop_duplicates(
        "mutation", keep="first"
    )
    frame = frame.rename(columns={resolved_score_col: rename_to})
    wt, pos, mut = zip(*[parse_mutation(token) for token in frame["mutation"]])
    frame["wt"] = wt
    frame["pos"] = pos
    frame["mut"] = mut
    return frame


def norm_cols(df, cols: Iterable[str], method: str):
    import numpy as np
    import pandas as pd

    output: Dict[str, Any] = {}
    if method == "zscore":
        for col in cols:
            values = df[col].astype(float).values
            mean = np.nanmean(values)
            std = np.nanstd(values) or 1.0
            output[col] = (values - mean) / std
    elif method == "robust_median_iqr":
        for col in cols:
            values = df[col].astype(float).values
            median = np.nanmedian(values)
            q25, q75 = np.nanpercentile(values, [25, 75])
            iqr = (q75 - q25) or 1.0
            output[col] = (values - median) / iqr
    elif method == "rank_percentile":
        for col in cols:
            output[col] = (
                pd.Series(df[col].astype(float).values)
                .rank(method="average", pct=True)
                .values
            )
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return output


def resolve_torch_device(device_mode: str) -> Tuple[Any, str]:
    import torch

    if device_mode == "cuda":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_mode == "cpu":
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device_name = (
        torch.cuda.get_device_name(0)
        if device == "cuda" and torch.cuda.is_available()
        else "CPU"
    )
    print(f"Device: {device.upper()} ({device_name})")
    return torch, device


def install_blosum_kernel_shim():
    import numpy as np
    from sklearn.gaussian_process.kernels import Kernel

    aa2idx = {aa: idx for idx, aa in enumerate(AA20)}
    blosum = np.zeros((20, 20), dtype=float)
    from Bio.Align import substitution_matrices

    raw = substitution_matrices.load("BLOSUM62")
    for aa_left in AA20:
        for aa_right in AA20:
            blosum[aa2idx[aa_left], aa2idx[aa_right]] = float(raw[aa_left, aa_right])
    blosum = (blosum - blosum.min()) / (blosum.max() - blosum.min())

    class BlosumKernel(Kernel):
        def __init__(self, alpha: float = 1.0):
            self.alpha = float(alpha)

        def __call__(self, X, Y=None, eval_gradient: bool = False):
            X = np.asarray(X, dtype=int)
            Y = X if Y is None else np.asarray(Y, dtype=int)
            sim_orig = blosum[X[:, 0][:, None], Y[:, 0][None, :]]
            sim_mut = blosum[X[:, 2][:, None], Y[:, 2][None, :]]
            same_pos = (X[:, 1][:, None] == Y[:, 1][None, :]).astype(float)
            kernel = self.alpha * 0.5 * (sim_orig + sim_mut) * same_pos
            if eval_gradient:
                return kernel, np.zeros((X.shape[0], Y.shape[0], 1))
            return kernel

        def diag(self, X):
            return np.full(len(X), self.alpha)

        def is_stationary(self) -> bool:
            return False

    globals()["BlosumKernel"] = BlosumKernel
    setattr(sys.modules["__main__"], "BlosumKernel", BlosumKernel)
    return aa2idx


def run_gpr_stage(config: Dict[str, Any], all_mutants, results_dir: str):
    import numpy as np
    import pandas as pd

    output_path = os.path.join(results_dir, "gpr_all.csv")
    if config["use_gpr_csv"]:
        print("Using provided GPR CSV override.")
        frame = try_read_csv(config["gpr_csv_path"])
        gpr_df = standardize_mutation_columns(frame, "GPR_score", "GPR_score")
        save_csv(gpr_df, output_path)
        return gpr_df, output_path

    if config["resume_runs"] and os.path.exists(output_path):
        print(f"Resuming existing GPR output: {output_path}")
        return pd.read_csv(output_path), output_path

    from joblib import load as joblib_load

    aa2idx = install_blosum_kernel_shim()
    print("Loading GPR model via joblib.")
    gpr = joblib_load(config["gpr_model_path"])

    features = np.array(
        [
            [aa2idx[row.wt], int(row.pos), aa2idx[row.mut]]
            for row in all_mutants.itertuples(index=False)
        ],
        dtype=int,
    )
    predictions = gpr.predict(features)

    gpr_df = all_mutants[["mutation", "wt", "pos", "mut"]].copy()
    gpr_df["GPR_score"] = predictions.astype(float)
    save_csv(gpr_df, output_path)
    return gpr_df, output_path


def run_esm2_stage(config: Dict[str, Any], wt_sequence: str, results_dir: str):
    import pandas as pd
    from tqdm.auto import tqdm

    output_path = os.path.join(results_dir, "esm2_all.csv")
    if config["use_esm2_csv"]:
        print("Using provided ESM-2 CSV override.")
        frame = try_read_csv(config["esm2_csv_path"])
        esm2_df = standardize_mutation_columns(frame, "ESM2_score", "ESM2_score")
        save_csv(esm2_df, output_path)
        return esm2_df, output_path

    if config["resume_runs"] and os.path.exists(output_path):
        print(f"Resuming existing ESM-2 output: {output_path}")
        return pd.read_csv(output_path), output_path

    torch, device = resolve_torch_device(config["device_mode"])
    import torch.nn.functional as F
    from transformers import EsmForMaskedLM, EsmTokenizer

    print(f"Loading ESM-2 model: {config['esm2_size']}")
    tokenizer = EsmTokenizer.from_pretrained(f"facebook/{config['esm2_size']}")
    model = EsmForMaskedLM.from_pretrained(f"facebook/{config['esm2_size']}")
    model = model.to(device).eval()

    encoded = tokenizer(wt_sequence, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    predictions = []
    total = len(wt_sequence) * 19
    pbar = tqdm(total=total, desc="ESM-2 scoring", unit="mut", miniters=1)

    for pos, wt in enumerate(wt_sequence, start=1):
        if pos >= input_ids.shape[1]:
            raise ValueError(
                f"Tokenized ESM-2 input is shorter than expected at position {pos}."
            )

        masked = input_ids.clone()
        masked[0, pos] = tokenizer.mask_token_id
        with torch.no_grad():
            logits = model(masked, attention_mask=attention_mask).logits[0, pos, :]

        if config["esm2_scoring_mode"] == "delta_logp_mut_minus_wt":
            log_probs = F.log_softmax(logits, dim=-1)
            wt_id = tokenizer.convert_tokens_to_ids(wt)
            if wt_id == tokenizer.unk_token_id:
                raise ValueError(f"Tokenizer could not resolve WT residue '{wt}'.")
        else:
            probs = F.softmax(logits, dim=-1)

        for aa in AA20:
            if aa == wt:
                continue
            mutation = f"{wt}{pos}{aa}"
            mut_id = tokenizer.convert_tokens_to_ids(aa)
            if mut_id == tokenizer.unk_token_id:
                raise ValueError(f"Tokenizer could not resolve mutant residue '{aa}'.")

            if config["esm2_scoring_mode"] == "delta_logp_mut_minus_wt":
                score = float((log_probs[mut_id] - log_probs[wt_id]).item())
            else:
                score = float(probs[mut_id].item())

            predictions.append((mutation, wt, pos, aa, score))
            pbar.set_postfix_str(f"mut={mutation}")
            pbar.update(1)

            if len(predictions) % int(config["checkpoint_every"]) == 0:
                checkpoint = pd.DataFrame(
                    predictions,
                    columns=["mutation", "wt", "pos", "mut", "ESM2_score"],
                )
                save_csv(checkpoint, output_path, to_print=False)

    pbar.close()
    esm2_df = pd.DataFrame(
        predictions,
        columns=["mutation", "wt", "pos", "mut", "ESM2_score"],
    )
    save_csv(esm2_df, output_path)
    return esm2_df, output_path


def run_stage1(config: Dict[str, Any], gpr_df, esm2_df, results_dir: str):
    import pandas as pd

    output_path = os.path.join(results_dir, f"stage1_top{config['top_k_mid']}.csv")

    stage1 = pd.merge(
        gpr_df[["mutation", "GPR_score"]],
        esm2_df[["mutation", "ESM2_score"]],
        on="mutation",
        how="inner",
    ).dropna()

    if stage1.empty:
        raise ValueError("Stage 1 produced no overlapping mutations to score.")

    normalized = norm_cols(stage1, ["GPR_score", "ESM2_score"], config["normalization"])
    stage1["score_stage1"] = (
        float(config["w_gpr_s1"]) * normalized["GPR_score"]
        + float(config["w_esm2_s1"]) * normalized["ESM2_score"]
    )

    stage1 = stage1.sort_values("score_stage1", ascending=False).reset_index(drop=True)
    limit = min(int(config["top_k_mid"]), len(stage1))
    stage1_top = stage1.head(limit).copy()

    wt, pos, mut = zip(*[parse_mutation(token) for token in stage1_top["mutation"]])
    stage1_top["wt"] = wt
    stage1_top["pos"] = pos
    stage1_top["mut"] = mut
    save_csv(stage1_top, output_path)
    return stage1_top, output_path


def load_structure_coords_and_seq(path: str, chain_id: str):
    import biotite.structure
    import numpy as np
    from biotite.sequence import ProteinSequence
    from biotite.structure import apply_residue_wise, get_chains
    from biotite.structure.io import pdb, pdbx
    from biotite.structure.residues import get_residues

    if not path:
        raise ValueError("pdb_path is empty.")

    biotite.structure.filter_backbone = biotite.structure.filter_peptide_backbone

    with open(path, "r", encoding="utf-8") as handle:
        if path.lower().endswith(".cif"):
            structure_file = pdbx.PDBxFile.read(handle)
            structure = pdbx.get_structure(structure_file, model=1)
        else:
            structure_file = pdb.PDBFile.read(handle)
            structure = pdb.get_structure(structure_file, model=1)

    mask = biotite.structure.filter_peptide_backbone(structure)
    structure = structure[mask]
    chains = get_chains(structure)
    if chain_id not in chains:
        raise ValueError(f"Chain '{chain_id}' not found; available chains: {chains}")
    structure = structure[[atom.chain_id == chain_id for atom in structure]]

    def extract(residue_atoms):
        atom_names = ["N", "CA", "C"]
        matrices = np.stack([residue_atoms.atom_name == name for name in atom_names], axis=1)
        indices = matrices.argmax(0)
        coords = residue_atoms[indices].coord
        coords[~matrices.any(0)] = np.nan
        return coords

    coords = apply_residue_wise(structure, structure, extract)
    _, residue_names = get_residues(structure)
    sequence = "".join(ProteinSequence.convert_letter_3to1(name) for name in residue_names)
    coord_mask = np.all(np.isfinite(coords), axis=(-1, -2))
    return coords, coord_mask, sequence


def ensure_seq_matches_pdb(wt_sequence: str, pdb_sequence: str) -> Tuple[str, bool]:
    same = wt_sequence == pdb_sequence
    if not same:
        print("WT sequence does not match the selected PDB chain sequence.")
        print(f"Using the PDB chain sequence for ESM-IF coordinates: len(WT)={len(wt_sequence)} len(PDB)={len(pdb_sequence)}")
    return pdb_sequence, same


def run_esmif_stage(config: Dict[str, Any], wt_sequence: str, stage1_top, results_dir: str):
    import pandas as pd

    output_path = os.path.join(results_dir, f"esmiF_top{config['top_k_mid']}.csv")
    if config["use_esmiF_csv"]:
        print("Using provided ESM-IF CSV override.")
        frame = try_read_csv(config["esmiF_csv_path"])
        esmif_df = standardize_mutation_columns(frame, "IF_score", "IF_score")
        save_csv(esmif_df, output_path)
        return esmif_df, output_path

    import numpy as np
    from tqdm.auto import tqdm

    torch, device = resolve_torch_device(config["device_mode"])
    import torch.nn.functional as F
    from esm.inverse_folding.util import CoordBatchConverter
    from esm.pretrained import esm_if1_gvp4_t16_142M_UR50

    coords, coord_mask, pdb_chain_seq = load_structure_coords_and_seq(
        config["pdb_path"], config["pdb_chain_id"]
    )
    if not np.any(coord_mask):
        raise ValueError("No finite backbone coordinates were found for the selected chain.")

    chain_seq_for_if, _ = ensure_seq_matches_pdb(wt_sequence, pdb_chain_seq)
    if config["resume_runs"] and os.path.exists(output_path):
        print(f"Resuming existing ESM-IF output: {output_path}")
        esmif_df = pd.read_csv(output_path)
    else:
        esmif_df = pd.DataFrame(columns=["mutation", "wt", "pos", "mut", "IF_score"])

    have = set(esmif_df["mutation"]) if not esmif_df.empty else set()
    todo = [row for row in stage1_top.itertuples(index=False) if row.mutation not in have]

    if todo:
        print(f"Loading ESM-IF model: {config['esm_if_variant']}")
        model, alphabet = esm_if1_gvp4_t16_142M_UR50()
        model = model.eval().to(device)
        batch_converter = CoordBatchConverter(alphabet)

        def score_sequence(sequence: str) -> float:
            batch = [(coords, None, sequence)]
            batch_coords, confidence, _, tokens, padding_mask = batch_converter(
                batch, device=device
            )
            previous_tokens = tokens[:, :-1]
            target = tokens[:, 1:]
            with torch.no_grad():
                logits, _ = model(batch_coords, padding_mask, confidence, previous_tokens)
                loss = F.cross_entropy(logits, target, reduction="none")[0]
            loss = loss.detach().float().cpu().numpy()
            return float(-np.sum(loss * coord_mask) / np.sum(coord_mask))

        predictions = []
        start = time.time()
        pbar = tqdm(total=len(todo), desc="ESM-IF scoring", miniters=1)
        for index, row in enumerate(todo, start=1):
            if int(row.pos) > len(chain_seq_for_if):
                raise ValueError(
                    f"Mutation {row.mutation} exceeds the selected PDB chain length."
                )
            mutated_sequence = apply_mutation(chain_seq_for_if, int(row.pos), row.mut)
            score = score_sequence(mutated_sequence)
            predictions.append((row.mutation, row.wt, int(row.pos), row.mut, score))

            elapsed = time.time() - start
            rate = index / elapsed if elapsed > 0 else float("inf")
            eta = (len(todo) - index) / rate if rate > 0 else -1
            pbar.set_postfix_str(f"mut={row.mutation} ETA={eta_str(eta)}")
            pbar.update(1)

            if index % int(config["checkpoint_every"]) == 0:
                partial = pd.DataFrame(
                    predictions,
                    columns=["mutation", "wt", "pos", "mut", "IF_score"],
                )
                esmif_df = pd.concat([esmif_df, partial], axis=0).drop_duplicates(
                    "mutation", keep="last"
                )
                save_csv(esmif_df, output_path, to_print=False)

        pbar.close()

        if predictions:
            partial = pd.DataFrame(
                predictions,
                columns=["mutation", "wt", "pos", "mut", "IF_score"],
            )
            esmif_df = pd.concat([esmif_df, partial], axis=0).drop_duplicates(
                "mutation", keep="last"
            )
            save_csv(esmif_df, output_path)
    else:
        print("All ESM-IF mutations were already scored in an existing result file.")

    return esmif_df, output_path


def run_final_stage(config: Dict[str, Any], stage1_top, esmif_df, results_dir: str):
    merged = stage1_top[["mutation", "GPR_score", "ESM2_score", "score_stage1"]].merge(
        esmif_df[["mutation", "IF_score"]],
        on="mutation",
        how="left",
    )
    missing_if = merged.loc[merged["IF_score"].isna(), "mutation"].tolist()
    if missing_if:
        raise ValueError(
            "Missing ESM-IF scores for final aggregation: "
            + ", ".join(missing_if[:10])
            + ("..." if len(missing_if) > 10 else "")
        )

    normalized = norm_cols(merged, ["GPR_score", "ESM2_score", "IF_score"], config["normalization"])
    merged["score_final"] = (
        float(config["w_gpr_final"]) * normalized["GPR_score"]
        + float(config["w_esm2_final"]) * normalized["ESM2_score"]
        + float(config["w_esmiF_final"]) * normalized["IF_score"]
    )
    merged = merged.sort_values("score_final", ascending=False).reset_index(drop=True)

    limit = min(int(config["top_k_final"]), len(merged))
    final_top = merged.head(limit).copy()
    wt, pos, mut = zip(*[parse_mutation(token) for token in final_top["mutation"]])
    final_top["wt"] = wt
    final_top["pos"] = pos
    final_top["mut"] = mut

    output_path = os.path.join(results_dir, f"EvoMax_final_top{config['top_k_final']}.csv")
    save_csv(final_top, output_path)
    return final_top, output_path


def run_pipeline(config: Dict[str, Any]) -> Dict[str, Any]:
    validate_config(config)
    results_dir = Path(config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Step 0/6 - Enumerating single substitutions")
    all_mutants = enumerate_single_mutants(config["wt_sequence"])
    all_mutants_path = str(results_dir / "all_single_mutants.csv")
    save_csv(all_mutants, all_mutants_path)
    print(f"Total mutants: {len(all_mutants):,} (L={len(config['wt_sequence'])} x 19)")

    print("Step 1/6 - Running GPR stage")
    gpr_df, gpr_path = run_gpr_stage(config, all_mutants, str(results_dir))

    print("Step 2/6 - Running ESM-2 stage")
    esm2_df, esm2_path = run_esm2_stage(config, config["wt_sequence"], str(results_dir))

    print("Step 3/6 - Combining Stage 1 scores")
    stage1_top, stage1_path = run_stage1(config, gpr_df, esm2_df, str(results_dir))

    print("Step 4/6 - Running ESM-IF stage")
    esmif_df, esmif_path = run_esmif_stage(
        config, config["wt_sequence"], stage1_top, str(results_dir)
    )

    print("Step 5/6 - Aggregating final ranking")
    final_top, final_path = run_final_stage(config, stage1_top, esmif_df, str(results_dir))

    print("Top final mutations:")
    print(final_top.head(20).to_string(index=False))
    print("\nArtifacts written:")
    print(f" - All mutants: {all_mutants_path}")
    print(f" - GPR predictions: {gpr_path}")
    print(f" - ESM-2 predictions: {esm2_path}")
    print(f" - Stage 1 top set: {stage1_path}")
    print(f" - ESM-IF scores: {esmif_path}")
    print(f" - Final ranking: {final_path}")

    return {
        "all_single_mutants_path": all_mutants_path,
        "gpr_path": gpr_path,
        "esm2_path": esm2_path,
        "stage1_path": stage1_path,
        "esmif_path": esmif_path,
        "final_path": final_path,
        "final_top_mutation": final_top.iloc[0]["mutation"],
        "final_row_count": int(len(final_top)),
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the EvoMax Docker-friendly pipeline.")
    parser.add_argument("--config", help="Path to a JSON config file.")
    parser.add_argument(
        "--print-default-config",
        action="store_true",
        help="Print the default config JSON and exit.",
    )
    args = parser.parse_args(argv)

    if not args.print_default_config and not args.config:
        parser.error("--config is required unless --print-default-config is used.")
    return args


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_default_config:
        print(json.dumps(DEFAULT_CONFIG, indent=2, sort_keys=True))
        return 0

    config = load_config(args.config)
    run_pipeline(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
