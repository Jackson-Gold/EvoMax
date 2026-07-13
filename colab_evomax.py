# -*- coding: utf-8 -*-
# Copyright (C) 2026 The Trustees of the University of Pennsylvania.
# Licensed under the Penn Software License (non-commercial). See LICENSE file.
"""Colab_EvoMax.ipynb

# 🧬 Colab EvoMax: Mutation Ranking Pipeline (GPR + ESM‑2 + ESM‑IF)

**One‑click Colab pipeline** to enumerate every single‑site mutation for an input sequence, score with a **Gaussian Process Regressor (GPR)** and **ESM‑2**, advance the **top 1.5%**, re‑score those with **ESM‑IF** + structure, and produce a **final Top‑100** ranking using configurable weights.

**Highlights**  
- Form fields (`#@title`) for a codeless UX  
- GPU **default** with CPU fallback (prints active device)  
- CSV overrides to **skip** any model run (GPR, ESM‑2, ESM‑IF)  
- Frequent **progress + ETA** messages, current mutation shown  
- **Resumability**: checkpoints to `/content/EvoMax/results`  

> For defaults not explicitly set here, logic mirrors the attached `.py/.ipynb` (GPR tuple features, ESM‑2 masked‑LM scoring, ESM‑IF inverse‑folding log‑likelihood, robust normalization, and weighting patterns).
"""

#@title ⬇️ EvoMax — Install dependencies (~8 min | run once per runtime)
# This cell installs runtime deps for ESM-2, ESM-IF, I/O, and progress reporting.
# It is safe to re-run; if Colab asks for a restart after torch-geometric, do so.

!pip -q install joblib tqdm biopython==1.84 pandas numpy
!pip -q install transformers
!pip -q install fair-esm biotite

# torch-geometric & torch-scatter are often required by ESM-IF on GPU.
# On Colab these usually "just work" on CPU and commonly work on GPU too.
# If you see an install mismatch warning, switch device to CPU in the form below.
!pip -q install torch-geometric torch-scatter

"""### Relevant Installed Packages and Versions

This cell checks and prints the versions of the key packages explicitly installed and utilized in this notebook.
"""

import pkg_resources

def get_version(package_name):
    try:
        return pkg_resources.get_distribution(package_name).version
    except pkg_resources.DistributionNotFound:
        return "Not Found"
    except Exception as e:
        return f"Error: {e}"


relevant_packages = [
    "joblib",
    "tqdm",
    "biopython",
    "pandas",
    "numpy",
    "transformers",
    "fair-esm",
    "biotite",
    "torch-geometric", # This might be listed as `torch-scatter` or similar within pkg_resources
    "torch-scatter"    # This might be listed as `torch-scatter` or similar within pkg_resources
]

print("Relevant Packages and their Versions:")
for pkg in relevant_packages:
    version = get_version(pkg)
    print(f"  {pkg}: {version}")

#@title 🧩 EvoMax — Inputs & runtime settings { form-width: "45%" }
#@markdown **Provide a protein sequence, structure path, and optional CSV overrides.**
#@markdown <br>Resumability is built-in; choose device and weights below.

import os, re, math, time, json, warnings
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

BASE_DIR    = "/content/EvoMax"
RESULTS_DIR = f"{BASE_DIR}/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Required inputs ─────────────────────────────────────────────────────────────
wt_sequence = "MKRSREDEPTHPPTNPSLAHGIIPFWDEYSQQVSDELWACSRDSFHEFNQYNNKGCTDRWFKFSQFTVIESKPVFDVPLNVHYSITENVAFDNSKKPPQLKKAKKNQKTPQKFQADKSLKIRLYPNEQERTTLNQWMGTARWIYNKCLEFTNKSKGVKKNKKNFRTFVVNNDNYQTENQWVVNTPYDVRDAAANELLTAFKTNFEKKKAGMIDKFMIRFRRKKDRKDHFVLHCKHWKKKTGLYSFIRNIKSAKPLPEELQYDSIIIKNKLNHYYLCIPQVLDIRGENQAPQHSGQVVALDPGVRTFQTTFDLNGYSTKWGSGGAERIGRLCCAYDKLQSKWSQPEVRHCKRYKYKRAGRRIQQKIRNIVDDLLKKLCLWLCRNYQVILLPSFETQKMVKKLHRRINSKTARKMLTWSHYRFKQRLLHKAREHPWTHIYIVNEAYTSKTCSCCGHVYTVGSSEVFRCPSCGSIFDRDINGARNILLRFLTTHRISF" #@param {type:"string"}
#@markdown *Paste the WT sequence (single-line, 20 canonical AAs only).*

pdb_path = "Round2Winner_0894a_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_000.pdb"    #@param {type:"string"}
#@markdown *Path to .pdb/.cif in this runtime or on Drive (e.g., `/content/file.pdb`). Upload in the next cell if needed.*

pdb_chain_id = "A"  #@param {type:"string"}

# ── Device selection ────────────────────────────────────────────────────────────
device_mode = "auto"  #@param ["auto", "cuda", "cpu"]
#@markdown *Default is “auto”: CUDA if available else CPU.*

# ── Stage sizes ────────────────────────────────────────────────────────────────
top_fraction_mid = 0.015  #@param {type:"number"}
#@markdown *The manuscript protocol advances the top 1.5% of all single-site candidates.*
top_k_mid = max(1, math.floor(len(wt_sequence) * 19 * top_fraction_mid))
top_k_final = 100  #@param {type:"integer"}

# ── GPR model (joblib) ─────────────────────────────────────────────────────────
gpr_model_path = "/content/GPR_BLOSUM.joblib"  #@param {type:"string"}
#@markdown *Will be loaded with `joblib.load(gpr_model_path)`.*

# ── Weights & normalization ────────────────────────────────────────────────────
# Stage‑1 (GPR + ESM‑2)
w_gpr_s1  = 0.35  #@param {type:"number"}
w_esm2_s1 = 0.65  #@param {type:"number"}

# Stage‑Final (GPR + ESM‑2 + ESM‑IF)
w_gpr_final  = 0.10  #@param {type:"number"}
w_esm2_final = 0.20  #@param {type:"number"}
w_esmiF_final= 0.70  #@param {type:"number"}

normalization = "robust_median_iqr"  #@param ["zscore", "robust_median_iqr", "rank_percentile"]
#@markdown *`robust_median_iqr` mirrors your notebook’s robust scaling approach.*

# ── CSV overrides to skip heavy steps ──────────────────────────────────────────
use_gpr_csv   = False  #@param {type:"boolean"}
gpr_csv_path  = ""     #@param {type:"string"}

use_esm2_csv  = False  #@param {type:"boolean"}
esm2_csv_path = ""     #@param {type:"string"}

use_esmiF_csv = False  #@param {type:"boolean"}
esmiF_csv_path= ""     #@param {type:"string"}

# ── ESM model choices ──────────────────────────────────────────────────────────
esm2_size = "esm2_t33_650M_UR50D"  #@param ["esm2_t33_650M_UR50D","esm2_t36_3B_UR50D","esm2_t48_15B_UR50D"]
#@markdown *650M is a practical default for Colab.*
esm_if_variant = "esm_if1_gvp4_t16_142M_UR50"  #@param ["esm_if1_gvp4_t16_142M_UR50"]

# ── Scoring & runtime ──────────────────────────────────────────────────────────
esm2_scoring_mode = "p_mut_only"   #@param ["p_mut_only","delta_logp_mut_minus_wt"]
#@markdown *Default mirrors your notebook (probability of mutant AA with masked position).*
checkpoint_every  = 20  #@param {type:"integer"}
resume_runs       = True  #@param {type:"boolean"}

# ── Validation ─────────────────────────────────────────────────────────────────
AA20 = "ACDEFGHIKLMNPQRSTVWY"
if wt_sequence:
    bad = sorted(set(ch for ch in wt_sequence) - set(AA20))
    if bad:
        raise ValueError(f"Found non-canonical residues in wt_sequence: {bad}. Only 20 AAs supported.")

#@title ⬆️ Upload PDB/CSV files (optional)
#@markdown Use this if you don’t already have your PDB/CSV in `/content`.
from google.colab import files
print("Select files to upload (e.g., .pdb, .cif, CSVs).")
_ = files.upload()
!ls -lh /content | sed -n '1,200p'

#@title 🛠️ EvoMax — Utility functions (no editing needed)
import joblib, torch, math, time, warnings
import torch.nn.functional as F

from typing import Tuple, List, Dict

# Device choice
if device_mode == "cuda":
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif device_mode == "cpu":
    device = "cpu"
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

dev_name = torch.cuda.get_device_name(0) if (device == "cuda" and torch.cuda.is_available()) else "CPU"
print(f"🖥️  Device: {device.upper()} {f'({dev_name})' if device=='cuda' else ''}")

def parse_mutation(tok:str)->Tuple[str,int,str]:
    m = re.fullmatch(r"([A-Z])(\d+)([A-Z])", tok)
    if not m: raise ValueError(f"Bad mutation token: {tok}")
    return m.group(1), int(m.group(2)), m.group(3)

def to_mutation(wt:str,pos:int,mut:str)->str:
    return f"{wt}{pos}{mut}"

def enumerate_single_mutants(seq:str)->pd.DataFrame:
    recs = []
    for i, wt in enumerate(seq, start=1):
        for aa in AA20:
            if aa == wt: continue
            recs.append({"mutation": f"{wt}{i}{aa}",
                        "wt": wt, "pos": i, "mut": aa})
    df = pd.DataFrame(recs)
    df["L"] = len(seq)
    return df

def apply_mutation(seq:str, pos:int, mut:str)->str:
    # pos is 1-based
    arr = list(seq)
    arr[pos-1] = mut
    return "".join(arr)

def norm_cols(df, cols, method="robust_median_iqr"):
    out = {}
    if method == "zscore":
        for c in cols:
            x = df[c].astype(float).values
            mu = np.nanmean(x); sd = np.nanstd(x) or 1.0
            out[c] = (x - mu) / sd
    elif method == "robust_median_iqr":
        for c in cols:
            x = df[c].astype(float).values
            med = np.nanmedian(x)
            q25, q75 = np.nanpercentile(x, [25, 75])
            iqr = (q75 - q25) or 1.0
            out[c] = (x - med) / iqr
    elif method == "rank_percentile":
        for c in cols:
            ranks = pd.Series(df[c].astype(float).values).rank(method="average", pct=True).values
            out[c] = ranks
    else:
        raise ValueError("Unknown normalization method.")
    return out

def eta_str(seconds: float) -> str:
    if seconds < 0: seconds = 0
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h: return f"{h}h {m}m {s}s"
    if m: return f"{m}m {s}s"
    return f"{s}s"

def save_csv(df, path, to_print = True):
    df.to_csv(path, index=False)
    if to_print:
      print(f"💾 Saved: {path}")

def try_read_csv(path, required=None):
    if not path or not os.path.exists(path): return None
    try:
        df = pd.read_csv(path)
        if required and not set(required).issubset(df.columns):
            print(f"⚠️  CSV {path} missing required columns {required}; ignoring.")
            return None
        return df
    except Exception as e:
        print(f"⚠️  Could not read {path}: {e}")
        return None

def standardize_mutation_columns(df, score_col, rename_to):
    df = df.copy()
    # If no 'mutation', try to build from wt/pos/mut
    if "mutation" not in df.columns:
        if all(c in df.columns for c in ["wt","pos","mut"]):
            df["mutation"] = df["wt"] + df["pos"].astype(int).astype(str) + df["mut"]
        else:
            raise ValueError(f"{score_col} CSV must contain 'mutation' or (wt,pos,mut).")
    if score_col not in df.columns:
        # common alternates
        for alt in ["score","pred_fold","ll","log_likelihood","prob"]:
            if alt in df.columns:
                score_col = alt
                break
    df = df[["mutation", score_col]].dropna().drop_duplicates("mutation", keep="first")
    df = df.rename(columns={score_col: rename_to})
    # Parse wt/pos/mut to help downstream joins
    wt,pos,mut = zip(*[parse_mutation(m) for m in df["mutation"]])
    df["wt"] = wt; df["pos"] = pos; df["mut"] = mut
    return df

#@title 🧠 EvoMax — Structure utilities (ESM-IF)
import numpy as np
from biotite.structure.io import pdb, pdbx
import biotite.structure
from biotite.structure import apply_residue_wise
from biotite.structure.residues import get_residues
from biotite.structure import get_chains
from biotite.sequence import ProteinSequence

# API patch used in various ESM-IF utilities
biotite.structure.filter_backbone = biotite.structure.filter_peptide_backbone

def load_structure_coords_and_seq(fpath, chain_id):
    if not fpath:
        raise ValueError("pdb_path is empty.")
    with open(fpath) as fin:
        if fpath.lower().endswith(".cif"):
            pdbxf = pdbx.PDBxFile.read(fin)
            struct = pdbx.get_structure(pdbxf, model=1)
        else:
            pdbf = pdb.PDBFile.read(fin)
            struct = pdb.get_structure(pdbf, model=1)

    mask = biotite.structure.filter_peptide_backbone(struct)
    struct = struct[mask]
    chains = get_chains(struct)
    if chain_id not in chains:
        raise ValueError(f"Chain '{chain_id}' not found; available: {chains}")
    struct = struct[[atom.chain_id == chain_id for atom in struct]]

    def extract(res_atoms):
        names = ["N","CA","C"]
        mats  = np.stack([res_atoms.atom_name == n for n in names], axis=1)
        idx   = mats.argmax(0)
        crd   = res_atoms[idx].coord
        crd[~mats.any(0)] = np.nan
        return crd

    coords = apply_residue_wise(struct, struct, extract)
    _, res_names = get_residues(struct)
    seq = "".join(ProteinSequence.convert_letter_3to1(r) for r in res_names)
    coord_mask = np.all(np.isfinite(coords), axis=(-1, -2))
    return coords, coord_mask, seq

def ensure_seq_matches_pdb(wt_seq: str, pdb_seq: str):
    if not wt_seq: return pdb_seq, True
    same = (wt_seq == pdb_seq)
    if not same:
        print("⚠️  WT sequence != PDB chain sequence. For ESM-IF we will use the PDB sequence for coordinates.")
        print(f"    len(WT)={len(wt_seq)} | len(PDB_chain)={len(pdb_seq)}")
    return pdb_seq, same

#@title 🧬 EvoMax — Enumerate single mutants (all 19 substitutions per position)
if not wt_sequence:
    raise ValueError("Please paste your wild-type sequence in the Inputs cell.")

print("Step 0/6 — Enumerating single substitutions…")
all_mutants = enumerate_single_mutants(wt_sequence)
print(f"Total mutants: {len(all_mutants):,} (L={len(wt_sequence)} × 19)")
save_csv(all_mutants, f"{RESULTS_DIR}/all_single_mutants.csv", True)

#@title 🤖 EvoMax — Stage‑1A: GPR predictions (~ 1 min)
# ─────────────────────── Make BlosumKernel available before loading ──────────────────────
import sys
import numpy as np
from sklearn.gaussian_process.kernels import Kernel
from Bio.SubsMat import MatrixInfo as matinfo

# Build AA → index map from existing AA20
AA2IDX = {aa: i for i, aa in enumerate(AA20)}

# Build normalized 20×20 BLOSUM62 matrix from your `matinfo.blosum62`
blo_raw = matinfo.blosum62
BLOSUM = np.zeros((20, 20), dtype=float)
for (a, b), score in blo_raw.items():
    if a in AA2IDX and b in AA2IDX:
        i, j = AA2IDX[a], AA2IDX[b]
        BLOSUM[i, j] = BLOSUM[j, i] = score
BLOSUM = (BLOSUM - BLOSUM.min()) / (BLOSUM.max() - BLOSUM.min())

class BlosumKernel(Kernel):
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.asarray(X, dtype=int)
        Y = X if Y is None else np.asarray(Y, dtype=int)
        sim_orig = BLOSUM[X[:, 0][:, None], Y[:, 0][None, :]]
        sim_mut  = BLOSUM[X[:, 2][:, None], Y[:, 2][None, :]]
        same_pos = (X[:, 1][:, None] == Y[:, 1][None, :]).astype(float)
        K = self.alpha * 0.5 * (sim_orig + sim_mut) * same_pos
        if eval_gradient:
            return K, np.zeros((X.shape[0], Y.shape[0], 1))
        return K

    def diag(self, X):
        return np.full(len(X), self.alpha)

    def is_stationary(self):
        return False

# Expose name on __main__ so unpickling resolves __main__.BlosumKernel
setattr(sys.modules['__main__'], 'BlosumKernel', BlosumKernel)

# ─────────────────────── Stage-1A: GPR predictions (joblib) with progress & ETA ──────────────────────
from joblib import load as joblib_load

gpr_out_path = f"{RESULTS_DIR}/gpr_all.csv"
if use_gpr_csv:
    print("Using provided GPR CSV override…")
    _df = try_read_csv(gpr_csv_path)
    if _df is None:
        raise ValueError("Could not read GPR override CSV.")
    gpr_df = standardize_mutation_columns(_df, score_col="GPR_score", rename_to="GPR_score")
    save_csv(gpr_df, gpr_out_path)
else:
    if resume_runs and os.path.exists(gpr_out_path):
        print(f"Resuming: found {gpr_out_path}")
        gpr_df = pd.read_csv(gpr_out_path)
    else:
        print("Loading GPR model via joblib…")
        gpr = joblib_load(gpr_model_path)  # per requirement
        AA2IDX = {aa:i for i,aa in enumerate(AA20)}

        preds = []
        t0 = time.time()
        total = len(all_mutants)
        pbar = tqdm(total=total, desc="GPR predicting", miniters=1)
        for i, row in enumerate(all_mutants.itertuples(index=False), start=1):
            X = np.array([[AA2IDX[row.wt], int(row.pos), AA2IDX[row.mut]]], dtype=int)
            y = float(gpr.predict(X)[0])
            preds.append((row.mutation, row.wt, int(row.pos), row.mut, y))
            # progress/ETA
            elapsed = time.time() - t0
            rate = i/elapsed if elapsed > 0 else float('inf')
            eta  = (total - i)/rate if rate > 0 else -1
            pbar.set_postfix_str(f"mut={row.mutation} ETA={eta_str(eta)}")
            pbar.update(1)
            if (i % checkpoint_every) == 0:
                gpr_df_tmp = pd.DataFrame(preds, columns=["mutation","wt","pos","mut","GPR_score"])
                save_csv(gpr_df_tmp, gpr_out_path, False)  # checkpoint
        pbar.close()

        gpr_df = pd.DataFrame(preds, columns=["mutation","wt","pos","mut","GPR_score"])
        save_csv(gpr_df, gpr_out_path, True)

print(gpr_df.head(3))

#@title 🧠 EvoMax — Stage‑1B: ESM‑2 (~ 30 min)

import os
import re
import time
import torch
import pandas as pd
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import EsmForMaskedLM, EsmTokenizer

# ----------------------------- Generate single-point mutations -----------------------------
standard_amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
single_point_mutations = []
for position, original_amino_acid in enumerate(wt_sequence, start=1):
    for new_amino_acid in standard_amino_acids:
        if new_amino_acid != original_amino_acid:
            mutated_sequence = list(wt_sequence)
            mutated_sequence[position - 1] = new_amino_acid
            mutated_sequence = "".join(mutated_sequence)
            mutation_string = f"{original_amino_acid}{position}{new_amino_acid}"
            single_point_mutations.append({
                'original_sequence': wt_sequence,
                'mutated_sequence': mutated_sequence,
                'mutation': mutation_string
            })

# ----------------------------- Output path & optional CSV override -----------------------------
esm2_out_path = f"{RESULTS_DIR}/esm2_all.csv"
if use_esm2_csv:
    _df = try_read_csv(esm2_csv_path)
    if _df is None:
        raise ValueError("Could not read ESM-2 override CSV.")
    esm2_df = standardize_mutation_columns(_df, score_col="ESM2_score", rename_to="ESM2_score")
    save_csv(esm2_df, esm2_out_path, True)
else:
    # ----------------------------- Resume from existing CSV if requested -----------------------------
    if resume_runs and os.path.exists(esm2_out_path):
        esm2_df = pd.read_csv(esm2_out_path)
    else:
        # ----------------------------- Load ESM-2 model & tokenizer -----------------------------
        tokenizer = EsmTokenizer.from_pretrained(f"facebook/{esm2_size}")
        model = EsmForMaskedLM.from_pretrained(f"facebook/{esm2_size}").to(device).eval()

        # ----------------------------- Scoring loop with tqdm + checkpointing -----------------------------
        preds = []
        t0 = time.time()
        total = len(single_point_mutations)
        pbar = tqdm(single_point_mutations, total=total, desc="ESM-2 scoring", unit="mut", miniters=1)

        for i, row in enumerate(pbar, start=1):
            mutated_seq = row['mutated_sequence']
            mutation = row['mutation']

            # show current mutation on the bar
            pbar.set_postfix_str(f"mut={mutation}")

            # Parse mutation string → wt, pos, mut
            m = re.match(r"([A-Z])(\d+)([A-Z])", mutation)
            if not m:
                continue
            wt, pos_str, mut = m.groups()
            pos = int(pos_str)

            # Tokenize mutated sequence and move to device
            enc = tokenizer(mutated_seq, return_tensors='pt').to(device)
            input_ids = enc['input_ids']
            attention_mask = enc['attention_mask']

            # In this setup, token index aligns with 1-based pos
            idx = pos

            # Bounds check
            if idx >= input_ids.shape[1]:
                continue

            # Mask the mutated position (ESM-2 MLM scoring)
            masked = input_ids.clone()
            masked[0, idx] = tokenizer.mask_token_id

            with torch.no_grad():
                logits = model(masked, attention_mask=attention_mask).logits

            # Probabilities at the mutated position
            probs = F.softmax(logits[0, idx, :], dim=-1)

            # Token id for the mutant residue
            mut_id = tokenizer.convert_tokens_to_ids(mut)
            if mut_id == tokenizer.unk_token_id:
                continue

            # Score (probability of mutant at the site)
            s = float(probs[mut_id].item())

            # Append in Stage-1B layout
            preds.append((mutation, wt, pos, mut, s))

            # Checkpoint every `checkpoint_every`
            if (i % checkpoint_every) == 0:
                esm2_df_tmp = pd.DataFrame(preds, columns=["mutation","wt","pos","mut","ESM2_score"])
                save_csv(esm2_df_tmp, esm2_out_path, False)

        pbar.close()

        # ----------------------------- Finalize DataFrame and save -----------------------------
        esm2_df = pd.DataFrame(preds, columns=["mutation","wt","pos","mut","ESM2_score"])
        save_csv(esm2_df, esm2_out_path, True)

#@title 🧮 EvoMax — Stage‑1C: Combine GPR+ESM‑2 → weighted score → Top‑K
stage1_path = f"{RESULTS_DIR}/stage1_top{top_k_mid}.csv"

s1 = pd.merge(gpr_df[["mutation","GPR_score"]],
              esm2_df[["mutation","ESM2_score"]],
              on="mutation", how="inner")
s1 = s1.dropna()

norm = norm_cols(s1, ["GPR_score","ESM2_score"], method=normalization)
s1["score_stage1"] = w_gpr_s1*norm["GPR_score"] + w_esm2_s1*norm["ESM2_score"]

s1 = s1.sort_values("score_stage1", ascending=False).reset_index(drop=True)
s1_top = s1.head(top_k_mid).copy()
# Add wt/pos/mut for convenience
wt,pos,mut = zip(*[parse_mutation(m) for m in s1_top["mutation"]])
s1_top["wt"]=wt; s1_top["pos"]=pos; s1_top["mut"]=mut

print(s1_top.head(10))
save_csv(s1_top, stage1_path)

#@title 🧱 EvoMax — Stage‑2: ESM‑IF scoring on Top‑K mutations (with progress & ETA)
import torch
from esm.pretrained import esm_if1_gvp4_t16_142M_UR50
from esm.inverse_folding.util import CoordBatchConverter
import torch.nn.functional as F
import numpy as np

esmiF_out_path = f"{RESULTS_DIR}/esmiF_top{top_k_mid}.csv"

if use_esmiF_csv:
    print("Using provided ESM‑IF CSV override…")
    _df = try_read_csv(esmiF_csv_path)
    if _df is None:
        raise ValueError("Could not read ESM‑IF override CSV.")
    esmiF_df = standardize_mutation_columns(_df, score_col="IF_score", rename_to="IF_score")
    save_csv(esmiF_df, esmiF_out_path)
else:
    coords, coord_mask, pdb_chain_seq = load_structure_coords_and_seq(pdb_path, pdb_chain_id)
    chain_seq_for_if, is_same = ensure_seq_matches_pdb(wt_sequence, pdb_chain_seq)

    if resume_runs and os.path.exists(esmiF_out_path):
        print(f"Resuming: found {esmiF_out_path}")
        esmiF_df = pd.read_csv(esmiF_out_path)
    else:
        print(f"Loading ESM‑IF model: {esm_if_variant} on {device.upper()} …")
        model, alphabet = esm_if1_gvp4_t16_142M_UR50()
        model = model.eval().to(device)
        batch_converter = CoordBatchConverter(alphabet)

        def score_seq_if(coords, seq: str) -> float:
            batch = [(coords, None, seq)]
            c, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)
            prev_tok = tokens[:, :-1]
            target   = tokens[:, 1:]
            with torch.no_grad():
                logits, _ = model(c, padding_mask, confidence, prev_tok)
                loss = F.cross_entropy(logits, target, reduction="none")[0]
            loss = loss.detach().float().cpu().numpy()
            ll_masked = -np.sum(loss * coord_mask) / np.sum(coord_mask)  # avg over residues with coords
            return float(ll_masked)

        # Resume-aware loop
        have = set()
        esmiF_df = pd.DataFrame(columns=["mutation","wt","pos","mut","IF_score"])
        if resume_runs and os.path.exists(esmiF_out_path):
            esmiF_df = pd.read_csv(esmiF_out_path)
            have = set(esmiF_df["mutation"])

        preds = []
        t0 = time.time()
        todo = [r for r in s1_top.itertuples(index=False) if r.mutation not in have]
        total = len(todo)
        if total == 0:
            print("All ESM‑IF mutations already scored from a previous run.")
        pbar = tqdm(total=total, desc="ESM‑IF scoring", miniters=1)
        for i, row in enumerate(todo, start=1):
            # Build mutant *on the PDB chain sequence*
            seq_for_if = apply_mutation(chain_seq_for_if, int(row.pos), row.mut)
            score = score_seq_if(coords, seq_for_if)
            preds.append((row.mutation, row.wt, int(row.pos), row.mut, score))
            # Progress
            elapsed = time.time() - t0
            rate = i/elapsed if elapsed > 0 else float('inf')
            eta  = (total - i)/rate if rate > 0 else -1
            pbar.set_postfix_str(f"mut={row.mutation} ETA={eta_str(eta)}")
            pbar.update(1)
            if (i % checkpoint_every) == 0:
                part = pd.DataFrame(preds, columns=["mutation","wt","pos","mut","IF_score"])
                esmiF_df = pd.concat([esmiF_df, part], axis=0).drop_duplicates("mutation", keep="last")
                save_csv(esmiF_df, esmiF_out_path)
        pbar.close()

        if len(preds):
            part = pd.DataFrame(preds, columns=["mutation","wt","pos","mut","IF_score"])
            esmiF_df = pd.concat([esmiF_df, part], axis=0).drop_duplicates("mutation", keep="last")

        save_csv(esmiF_df, esmiF_out_path)

print(esmiF_df.head(5))

#@title 🏁 EvoMax — Final re-aggregation & ranking (Top‑K output)
final_out_path = f"{RESULTS_DIR}/EvoMax_final_top{top_k_final}.csv"

# Join Stage‑1 and IF
merged = (s1_top[["mutation","GPR_score","ESM2_score","score_stage1"]]
          .merge(esmiF_df[["mutation","IF_score"]], on="mutation", how="left"))

# Normalize & weight (final)
normF = norm_cols(merged, ["GPR_score","ESM2_score","IF_score"], method=normalization)
merged["score_final"] = (
    w_gpr_final  * normF["GPR_score"] +
    w_esm2_final * normF["ESM2_score"] +
    w_esmiF_final* normF["IF_score"]
)

merged = merged.sort_values("score_final", ascending=False).reset_index(drop=True)
final_top = merged.head(top_k_final).copy()

# Add parsed columns for readability
wt,pos,mut = zip(*[parse_mutation(m) for m in final_top["mutation"]])
final_top["wt"]=wt; final_top["pos"]=pos; final_top["mut"]=mut

print("✅ Final Top candidates:")
try:
    from IPython.display import display
    display(final_top.head(20))
except Exception as _:
    print(final_top.head(20).to_string(index=False))

save_csv(final_top, final_out_path)

print("\nArtifacts written:")
print(f" • All mutants           → {RESULTS_DIR}/all_single_mutants.csv")
print(f" • GPR predictions       → {RESULTS_DIR}/gpr_all.csv")
print(f" • ESM‑2 predictions     → {RESULTS_DIR}/esm2_all.csv")
print(f" • Stage‑1 Top‑{top_k_mid}   → {RESULTS_DIR}/stage1_top{top_k_mid}.csv")
print(f" • ESM‑IF scores (Top‑K) → {RESULTS_DIR}/esmiF_top{top_k_mid}.csv")
print(f" • FINAL Top‑{top_k_final}   → {RESULTS_DIR}/EvoMax_final_top{top_k_final}.csv")
