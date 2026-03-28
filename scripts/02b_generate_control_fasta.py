"""
Generate a control FASTA file from introns of a specified gene list.

Used to produce the TNF-response gene control set for enrichment analyses
(Step 3). Reproduces the logic from make_bed_make_fasta_v2.ipynb:
  GTF → intron coords → sequence fetch → strand-aware output → FASTA

The output FASTA should be placed (or symlinked) at:
    <experiment_dir>/fasta_files/tnf_controls_peaks.fa

Then activate the comparison in your config:
    enrichment:
      comparisons:
        - primary: "all"
          control: "tnf_controls"

Usage
-----
    python scripts/generate_control_fasta.py --config config/my_run.yaml \\
        --gene-list manuscript_runs/tnf_gene_list.txt \\
        --output /path/to/fasta_files/tnf_controls_peaks.fa

    # Or use the built-in TNF gene list (LUKE_GENE_LIST):
    python scripts/generate_control_fasta.py --config config/my_run.yaml \\
        --use-tnf-list \\
        --output /path/to/fasta_files/tnf_controls_peaks.fa
"""
import os
import sys
import argparse
import logging
import yaml
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utility_functions import load_and_process_gtf, select_longest_basic_transcripts
from utils.sequence_functions import create_introns_dataframe, fetch_sequence, make_input_interval

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TNF-response gene list (LUKE_GENE_LIST from run_cosi.py)
# These are genes elicited in response to TNFa stimulation and represent
# the control set used in the manuscript enrichment analyses.
# ---------------------------------------------------------------------------
TNF_GENE_LIST = [
    'AW112010', 'Kdm6b', 'Cd83', 'Irg1', 'Clec4e', 'Traf1', 'Serpine1', 'Il12b',
    'Phf11a', 'Junb', 'Gpr85', 'Gpr84', 'Ripk2', 'Hivep2', 'Phrf1', 'Irf7', 'Slc16a10',
    'Irf1', 'Frmd6', 'Srgn', 'Cd69', 'Errfi1', 'Malt1', 'Gm14446', 'Cxcl1', 'Cxcl2',
    'Klhl25', 'Ccrn4l', 'Cxcl9', 'Nlrp3', 'Ifit3', 'Ifit2', 'Ifit1', 'Clic4', 'Tlr2',
    'Tnip3', 'Gnb4', 'Dpep3', 'Csf1', 'Snx18', 'Jdp2', 'Gdf15', 'Trib1', 'Ccnd2', 'Irs2',
    'Gem', 'Gm889', 'Ms4a4c', 'Maff', 'Kctd12', 'Gbp4', 'Gbp3', 'Ell2', 'Gbp2', 'Pde4b',
    'Odc1', 'Fnbp1l', 'Nfkbia', 'Abhd17c', 'Fmnl2', 'Il6', 'Nfkbie', 'Nfkbid', 'Nfkbib',
    'Sirt2', 'Phlda1', 'Fabp4', 'Iigp1', 'Stx11', 'Olfr56', 'Ccl2', 'Treh', 'Novel',
    'Ifi44', 'Oasl1', 'Rasgef1b', 'Ccl5', 'Ccl4', 'Ccl7', 'Nfkbiz', 'Zswim4', 'Bcl2l11',
    'Fam26f', 'Rnf19a', 'Peli1', 'Rnf19b', 'Ptgs2', 'Icosl', 'Nr4a1', 'Gm4951', 'Cfb', 'C2',
    'Zbtb12', 'Ier2', 'Ier3', 'Tfec', 'Itgav', 'C3', 'Tma16', 'Ccl12', 'Dhx40', 'Rab11fip1',
    'Ets2', 'Osgin2', 'Foxp4', 'Agpat4', 'Rcan1', 'Ch25h', 'Tnfsf9', 'Skil', 'Niacr1', 'Atf5',
    'Nup62-il4i1', 'Nup62', 'Il4i1', 'Tbc1d17', 'Daxx', 'Ifih1', 'Btg2', 'Ifi204', 'Ifi205',
    'Csrnp1', 'Extl2', 'Gm11435', 'Pyhin1', 'Agrn', 'Ccrl2', 'Ccdc146', 'Fgl2', 'Gm6377',
    'Zhx2', 'Fpr2', 'Tnf', 'Tnfaip3', 'Samsn1', 'Tnfaip2', 'Isg15', 'Klf7', 'Dusp1',
    'Slc28a2', 'Dusp2', 'Helz2', 'Dusp5', 'Plau', 'Dusp4', 'Dusp8', 'Sod2', 'Tnfsf15',
    'Isg20', 'Zbp1', '9930111J21Rik1', 'Tgtp1', '9930111J21Rik2', 'Tgtp2', 'Ifi47', 'Olfr1396',
    'Prdm1', 'Orai2', 'Edn1', 'Vcan', 'Cd274', 'Tnfsf10', 'Egr1', 'Egr2', 'Fam20c', 'Egr3',
    'Ifrd1', 'Plscr1', 'Jak2', 'Myc', 'Gm12250', 'Mapkapk2', 'Rsad2', 'Icam4', 'Fchsd2',
    'Rab20', 'Slc7a11', 'Icam1', 'Adamts1', 'Mid1', 'Pmaip1', 'Il1rn', 'Id3', 'Cd40',
    'Insig1', 'Atf3', 'Parp14', 'Slc44a1', 'Usp18', 'Snn', 'Txndc11', 'Plaur', 'Ppp1r15a',
    'Rgs1', 'Sdc4', 'Slco3a1', 'Slfn4', 'Igtp', 'Cish', 'Slfn1', 'AA467197', 'Plek', 'Cnrip1',
    'Ifnb1', 'Nod1', 'Nod2', 'Zc3h12c', 'Nos2', 'Zfp36', 'Sgms1', 'Cers6', 'Fos', 'Cmpk2',
    'Marcksl1', 'Art3', 'Cxcl11', 'Il1a', 'Cxcl10', 'Il1b', 'Cd44', 'Il15ra', 'Trim21',
    'Tgm2', 'Med21', 'Gm6654', 'Gbp5', 'Gbp9', 'Arl5c', 'Apobr', 'Il27', 'Saa3', 'Socs3',
    'Gapdh',
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate a control FASTA from a gene list")
    p.add_argument("--config",       required=True, help="Path to YAML config file")
    p.add_argument("--output",       required=True, help="Path for the output FASTA file")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--use-tnf-list", action="store_true",
                       help="Use the built-in TNF-response gene list (LUKE_GENE_LIST)")
    group.add_argument("--gene-list", metavar="FILE",
                       help="Path to a plain-text file with one gene name per line")
    p.add_argument("--flank", type=int, default=0,
                   help="Bp to extend each intron sequence on each side (default: 0)")
    p.add_argument("--min-intron-len", type=int, default=0,
                   help="Skip introns shorter than this many bp (default: 0)")
    return p.parse_args()


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _write_fasta(records: list, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    written = 0
    with open(output_path, "w") as f:
        for name, seq in records:
            if not seq:
                continue
            clean = str(name).replace(" ", "_").replace("\t", "_")
            f.write(f">{clean}\n{seq}\n")
            written += 1
    logger.info(f"Wrote {written} sequences to {output_path}")


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    gtf_file    = cfg.get("gtf_file")
    species     = cfg.get("species", "mouse")
    genome_name = cfg.get("genome_name")

    if not gtf_file or not os.path.exists(gtf_file):
        raise FileNotFoundError(f"gtf_file not found: {gtf_file}")

    # Load gene list
    if args.use_tnf_list:
        gene_list = TNF_GENE_LIST
        logger.info(f"Using built-in TNF gene list ({len(gene_list)} genes)")
    else:
        with open(args.gene_list) as f:
            gene_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(gene_list)} genes from {args.gene_list}")

    # Build intron coords from GTF
    logger.info("Loading and processing GTF...")
    gtf_df      = load_and_process_gtf(gtf_file, gene_list)
    gtf_df      = select_longest_basic_transcripts(gtf_df)
    introns_df  = create_introns_dataframe(gtf_df)

    if args.min_intron_len > 0:
        before = len(introns_df)
        introns_df = introns_df[introns_df["intron_length"] >= args.min_intron_len]
        logger.info(f"Filtered {before - len(introns_df)} introns shorter than {args.min_intron_len} bp")

    logger.info(f"Fetching sequences for {len(introns_df)} introns (flank={args.flank} bp)...")

    records = []
    n_failed = 0
    for _, row in introns_df.iterrows():
        chrom  = row["chrom"]
        start  = int(row["start"]) - args.flank
        end    = int(row["end"])   + args.flank
        strand = row["strand"]
        name   = row.get("intron_name",
                 f"{row.get('transcript_name', row.get('gene_name', 'unknown'))}_{row.get('intron_number', '')}")

        # Clamp to non-negative coordinates
        start = max(0, start)

        interval = pd.DataFrame({
            "chrom":  [chrom],
            "start":  [start],
            "end":    [end],
            "strand": [strand],
        })
        try:
            seq = fetch_sequence(interval, species, genome_version=genome_name)
            records.append((name, seq))
        except Exception as e:
            logger.warning(f"Failed to fetch sequence for {name} ({chrom}:{start}-{end}): {e}")
            n_failed += 1

    if n_failed:
        logger.warning(f"{n_failed} sequences failed to fetch and were skipped.")

    _write_fasta(records, args.output)
    logger.info("Done.")


if __name__ == "__main__":
    main()
