"""
Step 4 — Map MoDISco seqlet hits back to genomic coordinates and generate
per-motif, per-sequence logo + gene-map plots.

Usage
-----
    python scripts/04_map_motifs.py --config config/example_config.yaml

    # Override at the CLI:
    python scripts/04_map_motifs.py --config config/example_config.yaml \
        --measurement whole_transcript

Config keys read (under motif_mapping section, or top-level as fallback)
------------------------------------------------------------------------
    experiment_dir    : directory containing modisco run subdirectories
    flank             : flank value(s) used during MoDISco run (int or list) [default 50]
    measurement       : 'element_only' or 'whole_transcript'
    cosi_file_path    : coordinate CSV (same file used in steps 1-3)
    gtf_file          : GTF for gene-map visualisation
    modisco_window    : window (bp) used during MoDISco run  [default 1000]
    model_seq_len     : full Borzoi input length              [default 524288]
    selection_method  : 'modiscolite'                         [default]
    attr_respect_to   : 'element_only' or 'whole_transcript'  [default element_only]
    bigwig_dir        : optional directory of .bw coverage tracks
    motifs_of_interest: optional list of pattern labels to restrict
    motif_regex       : optional regex to filter pattern labels
    motif_limit       : optional integer cap on number of patterns plotted
    force_native_conversion : force recomputation of native-only attributions [default False]

Outputs (written per subset+flank under experiment_dir)
---------------------------------------------------------
    {subset}/masked_{flank}bp_flank/plots/indexing_df.csv
    {subset}/masked_{flank}bp_flank/plots/elements_df.csv
    {subset}/masked_{flank}bp_flank/plots/modiscolite/<pattern_label>/<seq_name>.png
    {subset}/masked_{flank}bp_flank/plots/modiscolite/<pattern_label>_hits.csv
"""
import os
import sys
import re
import time
import argparse
import logging
import pickle
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utility_functions import (
    load_and_process_gtf, load_meme, select_longest_basic_transcripts
)
from utils.sequence_functions import (
    create_introns_dataframe, create_elements_dataframe, attribution_native_only
)
from utils.helper_functions import handle_modiscolite

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULTS = {
    "model_seq_len":    524288,
    "modisco_window":   1000,
    "flank":            50,
    "plot":             {"modes": ["combined", "separate"]},
    "bigwig_dir":       None,
    "selection_method": "modiscolite",
    "attr_respect_to":  "element_only",
    "motifs_of_interest": None,
    "motif_regex":      None,
    "motif_limit":      None,
    "force_native_conversion": False,
}
MAPPING_SECTION = "motif_mapping"


def parse_args():
    p = argparse.ArgumentParser(description="Map MoDISco hits and generate plots")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--experiment_dir", help="Override experiment dir")
    p.add_argument("--output_dir", help="Override output dir")
    p.add_argument("--measurement", help="Override attr_respect_to / measurement")
    return p.parse_args()


def load_cfg(args) -> dict:
    with open(args.config) as f:
        raw = yaml.safe_load(f) or {}

    block = dict(raw.get(MAPPING_SECTION, {}))

    def pick(key, default=None):
        if key in block and block[key] is not None:
            return block[key]
        if key in raw and raw[key] is not None:
            return raw[key]
        return default

    cfg = {**DEFAULTS, **block}
    cfg["experiment_dir"]          = pick("experiment_dir")
    cfg["modisco_window"]          = pick("modisco_window", DEFAULTS["modisco_window"])
    cfg["flank"]                   = pick("flank", DEFAULTS["flank"])
    cfg["cosi_file_path"]          = pick("cosi_file_path")
    cfg["gtf_file"]                = pick("gtf_file")
    cfg["bigwig_dir"]              = pick("bigwig_dir", DEFAULTS["bigwig_dir"])
    cfg["attr_respect_to"]         = pick("attr_respect_to", DEFAULTS["attr_respect_to"])
    cfg["selection_method"]        = pick("selection_method", DEFAULTS["selection_method"])
    cfg["plot"]                    = {**DEFAULTS["plot"], **(pick("plot", {}) or {})}
    cfg["force_native_conversion"] = pick("force_native_conversion", DEFAULTS["force_native_conversion"])

    # CLI overrides
    if args.experiment_dir:
        cfg["experiment_dir"] = args.experiment_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.measurement:
        cfg["attr_respect_to"] = args.measurement

    # coord_file_path alias
    if not cfg.get("cosi_file_path") and pick("coord_file_path"):
        cfg["cosi_file_path"] = pick("coord_file_path")

    required = ["experiment_dir", "cosi_file_path", "gtf_file", "attr_respect_to"]
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise SystemExit(f"Missing required config field(s): {missing}")

    return cfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _tensor_window_start(seq_len: int, win: int) -> int:
    return (seq_len // 2) - (win // 2)


def _prep_indexing_df(cosi_csv, seq_len, modisco_window,
                       attr_respect_to="element_only", gtf_file=None):
    """
    Build the per-sequence indexing DataFrame used for coordinate mapping.

    If attr_respect_to == 'whole_transcript', transcript bounds are read from
    tscript_start/tscript_end columns in the CSV (preferred) or derived from
    the GTF (fallback).
    """
    df = pd.read_csv(cosi_csv)
    chrom_col = "chrom" if "chrom" in df.columns else "chr"

    idx = pd.DataFrame({
        "index":     df.index,
        "unique_ID": (df["unique_ID"].astype(str) if "unique_ID" in df.columns
                      else df["name"].astype(str)),
        "chrom":     df[chrom_col].astype(str),
        "start":     df["start"].astype(int),
        "end":       df["end"].astype(int),
        "strand":    df["strand"].astype(str),
    })

    if attr_respect_to == "whole_transcript":
        idx["element_start"] = idx["start"]
        idx["element_end"]   = idx["end"]

        if "tscript_start" in df.columns and "tscript_end" in df.columns:
            idx["start"] = df["tscript_start"].astype(int)
            idx["end"]   = df["tscript_end"].astype(int)
        else:
            if not gtf_file:
                raise ValueError(
                    "gtf_file must be provided when attr_respect_to == "
                    "'whole_transcript' and the CSV lacks tscript_start/tscript_end."
                )
            idx["gene_name"] = idx["unique_ID"].str.rsplit("_", n=1).str[0]
            gene_list = sorted({n.split("_")[0] for n in idx["unique_ID"].unique()})
            gtf_df = load_and_process_gtf(gtf_file, gene_list)
            gtf_df = select_longest_basic_transcripts(gtf_df)
            tx_df = (
                gtf_df.loc[gtf_df["feature"] == "transcript", ["gene_name", "start", "end"]]
                .sort_values(["gene_name", "start", "end"])
                .drop_duplicates(subset=["gene_name"], keep="first")
                .rename(columns={"start": "t_start", "end": "t_end"})
            )
            idx = idx.merge(tx_df, on="gene_name", how="left")
            have_tx = idx["t_start"].notna() & idx["t_end"].notna()
            idx.loc[have_tx, "start"] = idx.loc[have_tx, "t_start"].astype(int)
            idx.loc[have_tx, "end"]   = idx.loc[have_tx, "t_end"].astype(int)
            n_missing = (~have_tx).sum()
            if n_missing:
                logger.warning(f"{n_missing} rows missing transcript bounds in GTF; "
                                f"using element bounds for those rows.")
            idx = idx.drop(columns=["t_start", "t_end"])

    lengths = (idx["end"] - idx["start"]).astype(int)
    idx["tensor_start"]        = (seq_len // 2) - (lengths // 2)
    idx["tensor_window_start"] = _tensor_window_start(seq_len, modisco_window)
    return idx


def _subset_elements_for_indexing(gtf_file, indexing_df):
    """Build the elements_df (exons/introns/UTRs) for genes in indexing_df."""
    if "gene_name" in indexing_df.columns:
        gene_list = indexing_df["gene_name"].unique().tolist()
    else:
        gene_list = list({n.rsplit("_", 1)[0] for n in indexing_df["unique_ID"].unique()})
    gtf_df      = load_and_process_gtf(gtf_file, gene_list)
    gtf_df      = select_longest_basic_transcripts(gtf_df)
    introns_df  = create_introns_dataframe(gtf_df)
    elements_df = create_elements_dataframe(introns_df, gtf_df)
    return elements_df


def _maybe_load_bigwigs(bw_dir):
    if not bw_dir or not os.path.isdir(bw_dir):
        return {}
    bw = {fn[:-3]: os.path.join(bw_dir, fn)
          for fn in os.listdir(bw_dir) if fn.endswith(".bw")}
    return dict(sorted(bw.items()))


def _select_motifs(cfg, motifs):
    names = list(motifs.keys())
    if cfg.get("motifs_of_interest"):
        allow = [m for m in cfg["motifs_of_interest"] if m in motifs]
        if allow:
            names = allow
    if cfg.get("motif_regex"):
        rx    = re.compile(cfg["motif_regex"])
        names = [m for m in names if rx.search(m)]
    if cfg.get("motif_limit"):
        names = sorted(names)[:int(cfg["motif_limit"])]
    return names or list(motifs.keys())


def discover_modisco_subsets(cfg) -> list:
    """
    Discover modisco subset directories under experiment_dir.

    - RBP mode (multiple runs): directories ending in '_peaks_modisco'
    - Intron mode (single run): 'modisco' subdirectory
    - Fallback: '.' (experiment root) for legacy layouts
    """
    exp_dir = cfg["experiment_dir"]

    try:
        entries = os.listdir(exp_dir)
    except FileNotFoundError:
        raise SystemExit(f"Experiment dir not found: {exp_dir}")

    rbp_subsets = sorted(
        d for d in entries
        if d.endswith("_peaks_modisco") and os.path.isdir(os.path.join(exp_dir, d))
    )
    if rbp_subsets:
        logger.info(f"Discovered RBP subsets: {rbp_subsets}")
        return rbp_subsets

    if os.path.isdir(os.path.join(exp_dir, "modisco")):
        logger.info("Detected intron mode (found experiment_dir/modisco).")
        return ["modisco"]

    logger.warning("No RBP subsets or 'modisco' dir found; falling back to experiment root '.'")
    return ["."]


def _resolve_modisco_and_meme(cfg, subset: str, flank: int):
    """
    Resolve MoDISco H5 and motif file for a given subset + flank.

    Layout:
      - Intron: {EXP_DIR}/modisco/masked_{flank}bp_flank/...
      - RBP:    {EXP_DIR}/{subset}/masked_{flank}bp_flank/...
      - Legacy: {EXP_DIR}/masked_{flank}bp_flank/...  (subset == ".")
    """
    exp_dir   = cfg["experiment_dir"]
    flank_dir = f"masked_{flank}bp_flank"
    base_dir  = exp_dir if subset == "." else os.path.join(exp_dir, subset)

    candidates_h5 = [
        os.path.join(base_dir, flank_dir, "modisco_results.h5"),
        os.path.join(base_dir, flank_dir, "modisco_report.h5"),
    ]
    modisco_h5 = next((p for p in candidates_h5 if os.path.exists(p)), None)

    motif_file = None
    if modisco_h5:
        parent = os.path.dirname(modisco_h5)
        candidates_meme = [
            os.path.join(parent, "forward.meme"),
            os.path.join(parent, "meme.txt"),
        ]
        motif_file = next((p for p in candidates_meme if os.path.exists(p)), None)

    return modisco_h5, motif_file


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    cfg  = load_cfg(args)
    start_time = time.time()

    flank_cfg    = cfg.get("flank", 50)
    flank_values = [int(f) for f in flank_cfg] if isinstance(flank_cfg, list) else [int(flank_cfg)]

    bw_files = _maybe_load_bigwigs(cfg.get("bigwig_dir"))
    if bw_files:
        logger.info(f"Loaded {len(bw_files)} BigWig tracks")

    # Native-only attribution cache lives at experiment_dir level
    native_attr_path = os.path.join(cfg["experiment_dir"], "native_only_attributions.pkl")
    full_attr_path   = os.path.join(cfg["experiment_dir"], "attributions.pkl")

    subsets = discover_modisco_subsets(cfg)

    for subset in subsets:
        logger.info("=" * 80)
        logger.info(f"[SUBSET LOOP] Processing subset = {subset}")
        logger.info("=" * 80)

        for flank in flank_values:
            logger.info("-" * 80)
            logger.info(f"[subset={subset}] [flank={flank}] Running motif mapping")
            logger.info("-" * 80)

            modisco_h5, motif_file = _resolve_modisco_and_meme(cfg, subset, flank)

            if not modisco_h5 or not os.path.exists(modisco_h5):
                logger.warning(f"[subset={subset}][flank={flank}] Could not find MoDISco H5. Skipping.")
                continue

            if not motif_file or not os.path.exists(motif_file):
                logger.warning(f"[subset={subset}][flank={flank}] Could not find motif file. Skipping.")
                continue

            plot_root = os.path.join(os.path.dirname(modisco_h5), "plots")
            os.makedirs(plot_root, exist_ok=True)

            logger.info(f"[subset={subset}][flank={flank}] MoDISco H5:  {modisco_h5}")
            logger.info(f"[subset={subset}][flank={flank}] Motif file:  {motif_file}")
            logger.info(f"[subset={subset}][flank={flank}] Plot dir:    {plot_root}")

            motifs             = load_meme(motif_file)
            motifs_of_interest = _select_motifs(cfg, motifs)
            logger.info(f"Loaded {len(motifs)} motifs; selected {len(motifs_of_interest)}")

            # Indexing DataFrame (cached per plot_root)
            idx_df_path = os.path.join(plot_root, "indexing_df.csv")
            if os.path.exists(idx_df_path):
                idx_df = pd.read_csv(idx_df_path)
            else:
                idx_df = _prep_indexing_df(
                    cosi_csv=cfg["cosi_file_path"],
                    seq_len=int(cfg["model_seq_len"]),
                    modisco_window=int(cfg["modisco_window"]),
                    attr_respect_to=cfg["attr_respect_to"],
                    gtf_file=cfg["gtf_file"] if cfg["attr_respect_to"] == "whole_transcript" else None,
                )
                idx_df.to_csv(idx_df_path, index=False)

            # Elements DataFrame (cached per plot_root)
            elements_df_path = os.path.join(plot_root, "elements_df.csv")
            if os.path.exists(elements_df_path):
                elements_df = pd.read_csv(elements_df_path)
            else:
                elements_df = _subset_elements_for_indexing(cfg["gtf_file"], idx_df)
                elements_df.to_csv(elements_df_path, index=False)

            # Attributions — load cached native-only or convert and save
            if os.path.exists(native_attr_path) and not cfg.get("force_native_conversion"):
                logger.info(f"Loading cached native-only attributions from {native_attr_path}")
                final_attributions = _safe_load_pickle(native_attr_path)
            elif os.path.exists(full_attr_path):
                logger.info(f"Converting full attributions from {full_attr_path}")
                pt_attributions = _safe_load_pickle(full_attr_path)

                # Search multiple candidate paths for input_seqs.pkl
                seq_candidates = [
                    os.path.join(cfg["experiment_dir"], "input_seqs.pkl"),
                    os.path.normpath(os.path.join(cfg["experiment_dir"], "..", "..", "input_seqs.pkl")),
                ]
                input_seqs = None
                for p in seq_candidates:
                    if os.path.exists(p):
                        input_seqs = _safe_load_pickle(p)
                        break
                if input_seqs is None:
                    raise FileNotFoundError("Could not find input_seqs.pkl for native conversion.")

                final_attributions = np.array([
                    attribution_native_only(pt_attributions[i], seq)
                    for i, seq in enumerate(input_seqs)
                ])
                with open(native_attr_path, "wb") as f:
                    pickle.dump(final_attributions, f)
                logger.info(f"Saved native-only attributions to {native_attr_path}")
            else:
                raise FileNotFoundError(
                    f"Could not find attributions. Checked:\n"
                    f"  native: {native_attr_path}\n"
                    f"  full:   {full_attr_path}"
                )

            logger.info(f"[subset={subset}][flank={flank}] Attributions shape: {final_attributions.shape}")

            sel = str(cfg.get("selection_method", "modiscolite")).lower()
            logo_plot_dir = os.path.join(plot_root, sel)
            os.makedirs(logo_plot_dir, exist_ok=True)

            if sel == "modiscolite":
                handle_modiscolite(
                    h5_file=modisco_h5,
                    motifs_of_interest=motifs_of_interest,
                    indexing_df=idx_df,
                    pt_attributions=final_attributions,
                    logo_plot_dir=logo_plot_dir,
                    elements_df=elements_df,
                    MODEL_SEQ_LEN=int(cfg["model_seq_len"]),
                    MODISCO_WINDOW=int(cfg["modisco_window"]),
                    bw_files=bw_files,
                    y_scale=cfg.get("y_scale", "log1p"),
                    figsize=tuple(cfg.get("figsize", (20, 1.5))),
                    attr_respect_to=cfg.get("attr_respect_to", "element_only"),
                )
            else:
                logger.warning(
                    f"[subset={subset}][flank={flank}] Unknown selection_method '{sel}'. Skipping."
                )

    total_time = time.time() - start_time
    days, rem  = divmod(total_time, 86400)
    hours, rem = divmod(rem, 3600)
    mins, secs = divmod(rem, 60)
    logger.info(f"Total runtime: {int(days)}d {int(hours)}h {int(mins)}m {secs:.2f}s")


if __name__ == "__main__":
    main()
