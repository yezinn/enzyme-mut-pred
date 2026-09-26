"""
Zero-shot variant effect prediction on ProteinGym DMS benchmarks with ESM2.

Generalizes the pipeline validated on TEM-1 beta-lactamase to any ProteinGym
substitution assay, so that performance can be reported across multiple enzyme
families rather than a single protein.

Method: masked-marginal scoring (Meier et al., 2021, bioRxiv 2021.07.09.450648).
    LLR = log P(mutant residue | masked position) - log P(wild-type residue | masked position)

Engineering note: mutations that share a position share a forward pass, so the
model is run once per *position* (in batches) rather than once per mutation.
For a 346-residue protein with 6,227 mutations this is ~18x fewer forward passes.

Usage:
    python proteingym_benchmark.py --dms-id AMIE_PSEAE_Wrenbeck_2017
    python proteingym_benchmark.py --dms-id BLAT_ECOLX_Stiffler_2015 --model facebook/esm2_t33_650M_UR50D
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer, EsmForMaskedLM

# ProteinGym metadata and official benchmark results live in the GitHub repo
# (small files, no auth needed). The per-assay DMS data lives on HuggingFace /
# the Marks lab server and is fetched separately below.
PG_RAW = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main"
PG_REFERENCE_URL = f"{PG_RAW}/reference_files/DMS_substitutions.csv"
PG_BASELINE_URL = (
    f"{PG_RAW}/benchmarks/DMS_zero_shot/substitutions/Spearman/"
    "DMS_substitutions_Spearman_DMS_level.csv"
)

MUTANT_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
@dataclass
class AssayMetadata:
    dms_id: str
    dms_filename: str
    uniprot_id: str
    source_organism: str
    target_seq: str
    seq_len: int
    n_single_mutants: int
    selection_assay: str
    raw_phenotype: str
    raw_directionality: int


def load_assay_metadata(dms_id: str) -> AssayMetadata:
    """Fetch the official ProteinGym reference entry for one assay."""
    ref = pd.read_csv(PG_REFERENCE_URL)
    hits = ref[ref["DMS_id"] == dms_id]
    if hits.empty:
        available = ref["DMS_id"].tolist()
        close = [d for d in available if dms_id.split("_")[0] in d]
        raise ValueError(
            f"DMS_id '{dms_id}' not found in ProteinGym reference "
            f"({len(available)} assays). Similar ids: {close[:10]}"
        )
    r = hits.iloc[0]
    return AssayMetadata(
        dms_id=r["DMS_id"],
        dms_filename=r["DMS_filename"],
        uniprot_id=r["UniProt_ID"],
        source_organism=r["source_organism"],
        target_seq=str(r["target_seq"]),
        seq_len=int(r["seq_len"]),
        n_single_mutants=int(r["DMS_number_single_mutants"]),
        selection_assay=r["selection_assay"],
        raw_phenotype=r["raw_DMS_phenotype_name"],
        raw_directionality=int(r["raw_DMS_directionality"]),
    )


def official_baseline(dms_id: str, column: str = "ESM2 (650M)") -> Optional[float]:
    """Published Spearman for this assay from the ProteinGym leaderboard."""
    try:
        bench = pd.read_csv(PG_BASELINE_URL)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[warn] could not fetch ProteinGym baselines: {exc}")
        return None
    id_col = bench.columns[0]
    row = bench[bench[id_col] == dms_id]
    if row.empty or column not in bench.columns:
        return None
    return float(row.iloc[0][column])


# --------------------------------------------------------------------------
# DMS data download
# --------------------------------------------------------------------------
def download_dms_data(meta: AssayMetadata, cache_dir: str = "data") -> pd.DataFrame:
    """
    Download the per-variant DMS table for this assay.

    Tries the official HuggingFace mirror first (listing the repo so we do not
    have to guess the internal directory layout), then a community mirror.
    """
    os.makedirs(cache_dir, exist_ok=True)
    local = os.path.join(cache_dir, meta.dms_filename)
    if os.path.exists(local):
        print(f"[data] using cached {local}")
        return pd.read_csv(local)

    # 1) Official ProteinGym HuggingFace dataset repo.
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        repo_id = "OATML-Markslab/ProteinGym_v1"
        files = list_repo_files(repo_id, repo_type="dataset")
        matches = [f for f in files if f.endswith(meta.dms_filename)]
        if matches:
            path = hf_hub_download(repo_id, matches[0], repo_type="dataset")
            df = pd.read_csv(path)
            df.to_csv(local, index=False)
            print(f"[data] downloaded from HuggingFace: {matches[0]}")
            return df
        print(f"[warn] {meta.dms_filename} not found in {repo_id}; trying mirror")
    except Exception as exc:
        print(f"[warn] HuggingFace download failed ({exc}); trying mirror")

    # 2) Community mirror that stores single-substitution assays as TSV.
    mirror = (
        "https://huggingface.co/datasets/genbio-ai/ProteinGYM-DMS/resolve/main/"
        f"singles_substitutions/{meta.dms_id}.tsv"
    )
    try:
        df = pd.read_csv(mirror, sep="\t")
        df.to_csv(local, index=False)
        print(f"[data] downloaded from mirror: {mirror}")
        return df
    except Exception as exc:
        raise RuntimeError(
            "Could not download DMS data automatically.\n"
            f"  tried HuggingFace repo OATML-Markslab/ProteinGym_v1 and {mirror}\n"
            f"  last error: {exc}\n"
            "Fallback: download DMS_ProteinGym_substitutions.zip from "
            "https://proteingym.org (Downloads) and place "
            f"{meta.dms_filename} in ./{cache_dir}/"
        ) from exc


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def validate_mutants(df: pd.DataFrame, target_seq: str, mutant_col: str = "mutant") -> pd.DataFrame:
    """
    Keep single substitutions whose wild-type residue matches `target_seq`.

    ProteinGym numbers mutations 1-based against target_seq, but this is asserted
    rather than assumed: if a constant offset would fix a systematic mismatch, it
    is reported instead of silently producing meaningless scores.
    """
    singles = df[df[mutant_col].astype(str).str.match(MUTANT_RE)].copy()
    n_dropped = len(df) - len(singles)
    if n_dropped:
        print(f"[validate] dropped {n_dropped} non-single-substitution rows")

    parsed = singles[mutant_col].str.extract(MUTANT_RE)
    singles["wt_aa"] = parsed[0]
    singles["position"] = parsed[1].astype(int)
    singles["mut_aa"] = parsed[2]

    in_range = singles["position"].between(1, len(target_seq))
    if not in_range.all():
        raise ValueError(
            f"{(~in_range).sum()} mutations fall outside the target sequence "
            f"(length {len(target_seq)}); positions "
            f"{singles.loc[~in_range, 'position'].min()}–{singles.loc[~in_range, 'position'].max()}"
        )

    seq_aa = singles["position"].map(lambda p: target_seq[p - 1])
    match = seq_aa == singles["wt_aa"]
    if not match.all():
        rate = match.mean()
        raise ValueError(
            f"wild-type residue mismatch for {(~match).sum()}/{len(singles)} mutations "
            f"(match rate {rate:.1%}). The mutation numbering does not line up with "
            "target_seq — check the offset before trusting any scores."
        )
    print(f"[validate] all {len(singles)} single mutants match target_seq (offset = 1)")
    return singles


def check_score_direction(df: pd.DataFrame, meta: AssayMetadata, score_col: str = "DMS_score") -> None:
    """
    Confirm empirically that higher DMS_score means higher fitness.

    ProteinGym standardizes this, but the direction is verified from the data
    itself using the binary fitness label, because a silent sign flip would
    invert the reported correlation.
    """
    print(f"[direction] reference metadata: raw phenotype '{meta.raw_phenotype}', "
          f"raw_DMS_directionality = {meta.raw_directionality} "
          f"({'higher = fitter' if meta.raw_directionality == 1 else 'lower = fitter'} in the raw data)")

    if "DMS_score_bin" not in df.columns:
        print("[direction] no DMS_score_bin column; relying on ProteinGym's convention "
              "(higher DMS_score = fitter)")
        return

    fit = df.loc[df["DMS_score_bin"] == 1, score_col].mean()
    unfit = df.loc[df["DMS_score_bin"] == 0, score_col].mean()
    print(f"[direction] mean {score_col}: fit (bin=1) {fit:.3f} vs unfit (bin=0) {unfit:.3f}")
    if fit > unfit:
        print("[direction] OK — higher DMS_score = fitter, so a positive Spearman with "
              "LLR is the expected result (no sign flip).")
    else:
        raise ValueError(
            "DMS_score appears inverted (fit variants score lower than unfit ones). "
            "Flip the sign before computing Spearman."
        )


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@torch.no_grad()
def masked_marginal_scores(
    sequence: str,
    positions: list[int],
    model,
    tokenizer,
    device: str,
    batch_size: int = 16,
) -> dict[int, torch.Tensor]:
    """
    Log-probability vector at each requested 1-based position, with the position masked.

    One forward pass per position (batched), not one per mutation.
    """
    tokens = tokenizer(sequence, return_tensors="pt")
    base_ids = tokens["input_ids"][0]
    base_mask = tokens["attention_mask"][0]

    out: dict[int, torch.Tensor] = {}
    for start in range(0, len(positions), batch_size):
        chunk = positions[start : start + batch_size]
        ids = base_ids.unsqueeze(0).repeat(len(chunk), 1).clone()
        attn = base_mask.unsqueeze(0).repeat(len(chunk), 1)
        token_idx = []
        for row, pos in enumerate(chunk):
            idx = pos  # +1 for <cls> offsets the 1-based position
            ids[row, idx] = tokenizer.mask_token_id
            token_idx.append(idx)
        logits = model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits
        for row, pos in enumerate(chunk):
            out[pos] = torch.log_softmax(logits[row, token_idx[row]], dim=-1).cpu()
        print(f"\r[score] {min(start + batch_size, len(positions))}/{len(positions)} positions", end="")
    print()
    return out


def score_variants(df: pd.DataFrame, sequence: str, model, tokenizer, device: str,
                   batch_size: int = 16) -> pd.DataFrame:
    positions = sorted(df["position"].unique().tolist())
    print(f"[score] {len(df)} mutations span {len(positions)} positions "
          f"({len(df) / len(positions):.1f}x fewer forward passes than per-mutation scoring)")
    logprobs = masked_marginal_scores(sequence, positions, model, tokenizer, device, batch_size)

    aa_ids = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in set(df["wt_aa"]) | set(df["mut_aa"])}
    df = df.copy()
    df["esm2_llr"] = [
        (logprobs[p][aa_ids[m]] - logprobs[p][aa_ids[w]]).item()
        for p, w, m in zip(df["position"], df["wt_aa"], df["mut_aa"])
    ]
    return df


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dms-id", required=True, help="ProteinGym DMS id, e.g. AMIE_PSEAE_Wrenbeck_2017")
    ap.add_argument("--model", default="facebook/esm2_t33_650M_UR50D")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--data-dir", default="data")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = load_assay_metadata(args.dms_id)
    print(f"\n=== {meta.dms_id} ===")
    print(f"  {meta.uniprot_id} ({meta.source_organism}), {meta.seq_len} aa, "
          f"{meta.n_single_mutants} single mutants, assay: {meta.selection_assay}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    n_tokens = len(tokenizer(meta.target_seq)["input_ids"])
    limit = getattr(tokenizer, "model_max_length", 1024)
    limit = 1024 if limit is None or limit > 100000 else limit
    print(f"[tokens] sequence encodes to {n_tokens} tokens (model limit {limit})")
    if n_tokens > limit:
        raise SystemExit(
            f"Sequence is too long for {args.model} ({n_tokens} > {limit} tokens). "
            "Use a sliding-window approach or a model with a longer context."
        )

    df = download_dms_data(meta, args.data_dir)
    print(f"[data] {len(df)} rows, columns: {list(df.columns)}")
    df = validate_mutants(df, meta.target_seq)
    check_score_direction(df, meta)

    print(f"[model] loading {args.model} on {device}")
    model = EsmForMaskedLM.from_pretrained(args.model).to(device).eval()
    scored = score_variants(df, meta.target_seq, model, tokenizer, device, args.batch_size)

    rho, pval = spearmanr(scored["DMS_score"], scored["esm2_llr"])
    baseline = official_baseline(args.dms_id)

    print("\n--- result ---")
    print(f"  Spearman rho = {rho:.3f}  (p = {pval:.2e}, n = {len(scored)})")
    if baseline is not None:
        print(f"  ProteinGym published ESM2 (650M) baseline = {baseline:.3f}")
        print(f"  difference = {rho - baseline:+.3f}")
    else:
        print("  (no published baseline found for this assay/model)")

    out_csv = os.path.join(args.results_dir, f"{args.dms_id}_scored.csv")
    scored.to_csv(out_csv, index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 5))
        plt.scatter(scored["esm2_llr"], scored["DMS_score"], s=6, alpha=0.4)
        plt.xlabel("ESM2 zero-shot LLR (predicted mutation effect)")
        plt.ylabel("Experimental DMS score")
        title = f"{args.dms_id}\nSpearman rho = {rho:.3f}"
        if baseline is not None:
            title += f"  (ProteinGym ESM2-650M baseline: {baseline:.3f})"
        plt.title(title, fontsize=10)
        plt.tight_layout()
        fig_path = os.path.join(args.results_dir, f"{args.dms_id}_correlation.png")
        plt.savefig(fig_path, dpi=150)
        print(f"  saved {out_csv} and {fig_path}")
    except Exception as exc:
        print(f"  saved {out_csv} (plot skipped: {exc})")

    summary_path = os.path.join(args.results_dir, "benchmark_summary.csv")
    row = pd.DataFrame([{
        "dms_id": args.dms_id,
        "uniprot_id": meta.uniprot_id,
        "organism": meta.source_organism,
        "seq_len": meta.seq_len,
        "n_mutants": len(scored),
        "model": args.model,
        "spearman_rho": round(rho, 4),
        "p_value": pval,
        "proteingym_esm2_650M_baseline": baseline,
    }])
    if os.path.exists(summary_path):
        prev = pd.read_csv(summary_path)
        prev = prev[prev["dms_id"] != args.dms_id]
        row = pd.concat([prev, row], ignore_index=True)
    row.to_csv(summary_path, index=False)
    print(f"  updated {summary_path}")


if __name__ == "__main__":
    main()
