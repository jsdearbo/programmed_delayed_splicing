"""
Motif enrichment utilities: SEA (MEME Suite), FIMO (tangermeme),
MEME file I/O, and Modisco H5 conversion.
"""
import os
import logging
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List

import torch
import torch.nn.functional as F

try:
    from tangermeme.tools.fimo import fimo
except ImportError:
    fimo = None

from .utility_functions import load_meme

logger = logging.getLogger(__name__)


def check_dependencies():
    """Check for required external tools (tangermeme, MEME Suite sea)."""
    if fimo is None:
        logger.warning("tangermeme not found. FIMO functions will fail.")
    if subprocess.call("which sea", shell=True, stdout=subprocess.DEVNULL) != 0:
        logger.warning("MEME Suite 'sea' tool not found in PATH. SEA functions will fail.")


def _validate_dna_sequences(seqs: List[str]) -> List[str]:
    """Validate that sequences are DNA strings, not tensors/arrays."""
    valid = set("ACGTNacgtn")
    cleaned = []
    for i, s in enumerate(seqs):
        if not isinstance(s, str):
            raise TypeError(
                f"Sequence {i} is {type(s)}, expected str. "
                "You may be passing attribution arrays instead of sequences."
            )
        if s.strip().startswith("[") or (" " in s.strip() and not s.isalpha()):
            raise ValueError(
                f"Sequence {i} appears to be a stringified array: {s[:50]}..."
            )
        cleaned.append(s)
    return cleaned


def write_fasta(sequences: List[str], names: List[str], output_path: str):
    """Write sequences to a FASTA file with basic validation."""
    sequences = _validate_dna_sequences(sequences)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for name, seq in zip(names, sequences):
            if not seq:
                continue
            clean_name = str(name).replace(" ", "_").replace("\t", "_")
            f.write(f">{clean_name}\n{str(seq).strip()}\n")


# ---------------------------------------------------------------------------
# Motif Conversion
# ---------------------------------------------------------------------------

def reverse_complement_ppm(ppm: np.ndarray) -> np.ndarray:
    """Generate the reverse complement of a position probability matrix (PPM)."""
    complement_indices = [3, 2, 1, 0]  # A↔T, C↔G
    return ppm[complement_indices, ::-1]


def reverse_complement_motifs_dict(motifs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return a new dict with reverse-complement PPMs for every entry."""
    return {f"{key}_rc": reverse_complement_ppm(value) for key, value in motifs.items()}


def write_motifs_to_meme(motifs: Dict[str, Any], output_file: str,
                          background: Optional[Dict[str, float]] = None):
    """
    Write a motif dictionary to a MEME-format file.

    motifs values may be numpy arrays or torch tensors of shape (4, L) or (L, 4).
    """
    try:
        with open(output_file, "w") as f:
            f.write("MEME version 4\n\nALPHABET= ACGT\n\n")
            if background is None:
                background = {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25}
            bg_line = " ".join(f"{b} {background[b]}" for b in "ACGT")
            f.write(f"Background letter frequencies:\n{bg_line}\n\n")
            for name, ppm in motifs.items():
                if hasattr(ppm, 'numpy'):
                    ppm = ppm.numpy()
                elif isinstance(ppm, list):
                    ppm = np.array(ppm)
                if ppm.shape[0] == 4 and ppm.shape[1] != 4:
                    ppm = ppm.T
                L = ppm.shape[0]
                f.write(f"MOTIF {name}\n")
                f.write(f"letter-probability matrix: alength= 4 w= {L}\n")
                for i in range(L):
                    col = np.clip(ppm[i], 1e-6, 1.0)
                    col = col / col.sum()
                    f.write(" ".join(f"{v:.6f}" for v in col) + "\n")
                f.write("\n")
        logger.info(f"Motifs written to {output_file}")
    except Exception as e:
        logger.error(f"Failed to write MEME file: {e}")
        raise


def convert_modisco_h5_to_meme(report_file: str, output_dir: str):
    """
    Read a MoDISco HDF5 report and write forward.meme and combined.meme
    (forward + reverse-complement) to output_dir.
    """
    try:
        from grelu.io import motifs as grelu_motifs
    except ImportError:
        logger.error("grelu not found. Cannot read MoDISco H5 files.")
        return

    if not os.path.exists(report_file):
        logger.warning(f"MoDISco report not found at {report_file}. Skipping.")
        return

    try:
        logger.info(f"Loading motifs from {report_file}")
        modisco_output = grelu_motifs.read_modisco_report(report_file)
        rc_motifs = reverse_complement_motifs_dict(modisco_output)
        write_motifs_to_meme(modisco_output,
                              os.path.join(output_dir, "forward.meme"))
        write_motifs_to_meme({**modisco_output, **rc_motifs},
                              os.path.join(output_dir, "combined.meme"))
    except Exception as e:
        logger.error(f"Error converting MoDISco H5 to MEME: {e}")


# ---------------------------------------------------------------------------
# SEA (MEME Suite)
# ---------------------------------------------------------------------------

def run_sea(primary_fasta: str, control_fasta: str, meme_file: str,
             output_dir: str, thresh: float = 1.0e6) -> pd.DataFrame:
    """
    Run SEA (Simple Enrichment Analysis) from the MEME Suite.

    Returns the sea.tsv results as a DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "sea", "--verbosity", "1",
        "--oc", output_dir,
        "--thresh", str(thresh),
        "--align", "center",
        "--p", os.path.abspath(primary_fasta),
        "--n", os.path.abspath(control_fasta),
        "--m", os.path.abspath(meme_file),
    ]
    logger.info(f"Running SEA: {' '.join(cmd)}")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"SEA failed (exit {res.returncode}).\nSTDERR:\n{res.stderr}")
    tsv_path = os.path.join(output_dir, "sea.tsv")
    if not os.path.exists(tsv_path):
        raise FileNotFoundError(f"SEA completed but {tsv_path} not found.")
    return pd.read_csv(tsv_path, sep="\t")


# ---------------------------------------------------------------------------
# FIMO (tangermeme)
# ---------------------------------------------------------------------------

def _count_seqs_in_fasta(fasta_path: str) -> int:
    count = 0
    try:
        with open(fasta_path) as f:
            for line in f:
                if line.startswith(">"):
                    count += 1
    except Exception as e:
        logger.warning(f"Could not count sequences in {fasta_path}: {e}")
    return count


def _normalize_fimo_output(hits) -> pd.DataFrame:
    """Coerce FIMO output (list or DataFrame) to a single DataFrame."""
    if hits is None:
        return pd.DataFrame()
    if isinstance(hits, list):
        valid = [h for h in hits if isinstance(h, pd.DataFrame)]
        return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()
    if isinstance(hits, pd.DataFrame):
        return hits
    raise TypeError(f"Unexpected FIMO output type: {type(hits)}")


def _get_hit_percentage(fimo_hits, total_seqs: int) -> dict:
    """Return {motif_name: pct_sequences_with_hit} from FIMO output."""
    if total_seqs == 0:
        return {}
    fimo_df = _normalize_fimo_output(fimo_hits)
    if fimo_df.empty:
        return {}
    seq_col   = next((c for c in ["sequence_name", "sequence_id", "seq_name"]
                      if c in fimo_df.columns), None)
    motif_col = next((c for c in ["motif_name", "motif_id"]
                      if c in fimo_df.columns), None)
    if seq_col and motif_col:
        counts = fimo_df.groupby(motif_col)[seq_col].nunique()
        return {m: (n / total_seqs) * 100.0 for m, n in counts.items()}
    return {}


def run_fimo_enrichment(motifs: Dict[str, Any], primary_fasta: str,
                         control_fasta: str, output_dir: str,
                         threshold: float = 1e-4) -> pd.DataFrame:
    """
    Run FIMO on primary and control FASTA files and return per-motif
    hit percentages as a DataFrame.
    """
    if fimo is None:
        raise ImportError("tangermeme is required for FIMO enrichment.")

    validated_motifs = {}
    for name, motif in motifs.items():
        if isinstance(motif, np.ndarray):
            t = torch.tensor(motif, dtype=torch.float32)
        elif torch.is_tensor(motif):
            t = motif.float()
        else:
            t = torch.tensor(np.asarray(motif), dtype=torch.float32)
        if t.shape[0] != 4 and t.shape[1] == 4:
            t = t.T
        if t.shape[0] != 4:
            logger.warning(f"Skipping motif {name}: invalid shape {tuple(t.shape)}")
            continue
        validated_motifs[name] = t

    total_primary = _count_seqs_in_fasta(primary_fasta)
    total_control = _count_seqs_in_fasta(control_fasta)
    logger.info(f"FIMO: {total_primary} primary seqs, {total_control} control seqs")

    try:
        primary_results = fimo(motifs=validated_motifs,
                                sequences=os.path.abspath(primary_fasta),
                                reverse_complement=False, dim=0, threshold=threshold)
        pct_primary = _get_hit_percentage(primary_results, total_primary)

        control_results = fimo(motifs=validated_motifs,
                                sequences=os.path.abspath(control_fasta),
                                reverse_complement=False, dim=0, threshold=threshold)
        pct_control = _get_hit_percentage(control_results, total_control)
    except Exception as e:
        logger.error(f"FIMO run failed: {e}")
        return pd.DataFrame()

    all_motifs = set(pct_primary) | set(pct_control) | set(validated_motifs)
    rows = [
        {
            'motif_name':             m,
            'percent_match_primary':  pct_primary.get(m, 0.0),
            'percent_match_ctrl':     pct_control.get(m, 0.0),
        }
        for m in sorted(all_motifs)
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, "fimo_enrichment.csv"), index=False)
    logger.info(f"FIMO results saved to {output_dir}/fimo_enrichment.csv")
    return df


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_motif_scatter(df: pd.DataFrame,
                       x_col: str = 'percent_match_ctrl',
                       y_col: str = 'percent_match_primary',
                       label_col: str = 'motif_name',
                       title: str = "Motif Enrichment",
                       save_path: Optional[str] = None,
                       top_n: int = 10):
    """Scatter plot of primary vs control motif-hit percentages."""
    if df.empty:
        logger.warning("Empty DataFrame — nothing to plot.")
        return
    df = df.copy()
    df['log2fc'] = np.log2((df[y_col] + 0.1) / (df[x_col] + 0.1))
    plt.figure(figsize=(8, 8))
    sns.set_style("whitegrid")
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='log2fc',
                    palette='vlag', edgecolor='k', s=100, alpha=0.8)
    max_val = max(df[x_col].max(), df[y_col].max())
    plt.plot([0, max_val], [0, max_val], ls="--", c=".3")
    texts = []
    df_sorted = df.sort_values('log2fc', ascending=False)
    for i in range(min(top_n, len(df))):
        row = df_sorted.iloc[i]
        if row['log2fc'] > 0.5:
            texts.append(plt.text(row[x_col], row[y_col], row[label_col], fontsize=9))
    for i in range(min(top_n, len(df))):
        row = df_sorted.iloc[-(i + 1)]
        if row['log2fc'] < -0.5:
            texts.append(plt.text(row[x_col], row[y_col], row[label_col], fontsize=9))
    try:
        from adjustText import adjust_text
        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    except ImportError:
        pass
    plt.title(title)
    plt.xlabel("% Sequences with Motif (Control)")
    plt.ylabel("% Sequences with Motif (Primary)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        logger.info(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Main Orchestrator
# ---------------------------------------------------------------------------

def run_enrichment_analysis(primary_fasta: str, control_fasta: str,
                             meme_file: str, output_dir: str,
                             run_sea_tool: bool = True,
                             run_fimo_tool: bool = True):
    """
    Run SEA and/or FIMO enrichment analysis for one primary/control FASTA pair.

    Results are written under output_dir/sea/ and output_dir/fimo/.
    Skips any tool whose results already exist (safe to re-run).
    """
    logger.info(f"Starting enrichment analysis → {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    if run_sea_tool:
        sea_dir = os.path.join(output_dir, "sea")
        os.makedirs(sea_dir, exist_ok=True)
        if not os.path.exists(os.path.join(sea_dir, "sea.tsv")):
            try:
                sea_results = run_sea(primary_fasta, control_fasta, meme_file, sea_dir)
                sea_df = pd.DataFrame({
                    "motif_name":            sea_results["ID"],
                    "percent_match_primary": sea_results["TP%"],
                    "percent_match_ctrl":    sea_results["FP%"],
                })
                sea_df.to_csv(os.path.join(sea_dir, "sea_enrichment.csv"), index=False)
                plot_motif_scatter(
                    sea_df, title="SEA Enrichment: Primary vs Control",
                    save_path=os.path.join(sea_dir, "sea_enrichment_scatter.png")
                )
            except Exception as e:
                logger.error(f"SEA analysis failed: {e}")
        else:
            logger.info(f"SEA results already exist in {sea_dir}. Skipping.")

    if run_fimo_tool:
        fimo_dir = os.path.join(output_dir, "fimo")
        os.makedirs(fimo_dir, exist_ok=True)
        if not os.path.exists(os.path.join(fimo_dir, "fimo_enrichment.csv")):
            try:
                motifs = load_meme(meme_file)
                logger.info(f"Loaded {len(motifs)} motifs for FIMO.")
                fimo_df = run_fimo_enrichment(
                    motifs=motifs,
                    primary_fasta=primary_fasta,
                    control_fasta=control_fasta,
                    output_dir=fimo_dir,
                    threshold=0.001,
                )
                plot_motif_scatter(
                    fimo_df, title="FIMO Enrichment: Primary vs Control",
                    save_path=os.path.join(fimo_dir, "fimo_enrichment_scatter.png")
                )
            except Exception as e:
                logger.error(f"FIMO analysis failed: {e}")
        else:
            logger.info(f"FIMO results already exist in {fimo_dir}. Skipping.")
