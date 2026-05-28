# Programmed Delayed Splicing — Borzoi Fine-tuning and Splicing Motif Discovery Pipeline

Deep-learning pipeline for identifying sequence features associated with delayed
co-transcriptional splicing. Fine-tunes the [Borzoi](https://github.com/calico/borzoi)
sequence-to-function model (via [gReLU](https://github.com/Genentech/gReLU)) on
timecourse RNA-seq data, then computes per-nucleotide attributions over introns,
discovers *de novo* motifs with [TF-MoDISco](https://github.com/jmschrei/tfmodisco-lite),
and measures motif enrichment between regulatory subsets.

This pipeline reproduces the figures in:

> Dearborn J.S., Frankiw L., Limoge D.W., Burns C.H., Vlach L., Turpin P., Kirch T., Miller Z.D., Dowell W., Languon S., Garcia‑Flores Y., Cockrell R.C., Baltimore D., Majumdar D. *Programmed Delayed Splicing: A Mechanism for Timed Inflammatory Gene Expression*. eLife (2026). https://doi.org/10.7554/eLife.109726.1

---

## Overview

The pipeline runs in five sequential steps, all driven by a single YAML config file:

```
00_fine_tune.py  →  01_get_attributions.py  →  02_run_modisco.py  →  03_run_enrichment.py  →  04_map_motifs.py
      ↓                       ↓                        ↓                      ↓                       ↓
 checkpoint.ckpt         attributions.pkl         modisco.h5 +           sea.tsv +             logos +
                         input_seqs.pkl           forward.meme          fimo results           gene-map PNGs
```

| Step | Script | What it does |
|------|--------|--------------|
| 0 | `00_fine_tune.py` | Fine-tunes Borzoi on timecourse RNA-seq BigWig data |
| 1 | `01_get_attributions.py` | Computes input×gradient attributions over each intron |
| 2 | `02_run_modisco.py` | Masks attributions to intron ± flank, runs TF-MoDISco, exports motifs to MEME |
| 3 | `03_run_enrichment.py` | Compares motif enrichment between subsets using SEA and FIMO |
| 4 | `04_map_motifs.py` | Maps MoDISco seqlets back to genomic coordinates, generates logo + gene-map figures |

---

## Requirements

### System dependencies

- **MEME Suite** ≥ 5.5 (provides the `sea` command used in Step 3)
  Install: https://meme-suite.org/meme/doc/install.html
  Verify: `sea --version`

- **conda** or **mamba** (for environment setup)

### Reference genome

The pipeline fetches DNA sequences via [genomepy](https://github.com/vanheeringen-lab/genomepy),
which will automatically download the mm10 mouse genome (~800 MB) on first run:

```bash
genomepy install mm10
```

### Reference genome GTF

A GTF annotation file is required for Steps 3–4. The manuscript figures use
**GENCODE vM23 (GRCm38)**:

```bash
wget https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M23/gencode.vM23.annotation.gtf.gz
gunzip gencode.vM23.annotation.gtf.gz
```

Then set `gtf_file` in your config to the path of the uncompressed `.gtf` file.

### Python environment

```bash
conda env create -f environment.yml
conda activate elife-delay-splice
```

> **Note on gReLU / Borzoi weights.**
> `grelu` is installed from a custom fork. On first run, it will automatically
> download the pretrained Borzoi mouse weights (~2 GB). An internet connection
> is required for this step; subsequent runs use the cached weights.

---

## Installation

```bash
git clone https://github.com/jsdearbo/programmed_delayed_splicing.git
cd programmed_delayed_splicing
conda env create -f environment.yml
conda activate elife-delay-splice
```

No additional installation steps are required — the `utils/` package is imported
directly from the repository root by each script.

---

## Data preprocessing

The BigWig files consumed by Step 0 are derived from raw RNA-seq FASTQs through a two-step process. Scripts for both steps are in the `preprocessing/` directory. Raw FASTQs for the manuscript are deposited at GEO (accession to be added upon publication).

### Step A: FASTQ → BAM (STAR alignment)

```bash
bash preprocessing/star_loop_v2.sh
```

The script prompts interactively for species (mouse/human), the path to your STAR index directory, and the directory containing FASTQ files. It handles both paired-end (R1/R2) and single-end reads, and outputs coordinate-sorted BAMs.

**Dependencies:** STAR ≥ 2.7

### Step B: BAM → BigWig

```bash
bash preprocessing/bam_to_bigwig.sh /path/to/bam_dir [/path/to/bw_dir]
```

Or with options:

```bash
NORMALIZE=CPM BIN_SIZE=1 THREADS=8 STRANDED=none \
  bash preprocessing/bam_to_bigwig.sh /path/to/bam_dir /path/to/bw_dir
```

Key options (set as environment variables):

| Variable | Default | Description |
|----------|---------|-------------|
| `NORMALIZE` | `None` | Normalization method: `None`, `CPM`, `RPKM`, `BPM` |
| `BIN_SIZE` | `1` | Coverage bin size in bp |
| `THREADS` | `16` | Parallel threads |
| `SKIP_DUPLICATES` | `0` | Set to `1` to skip duplicate reads |
| `STRANDED` | `auto` | Strand mode: `none`, `forward`, `reverse` |
| `EXTEND` | `auto` | Fragment extension bp; `auto` estimates from read length for single-end |

**Dependencies:** deepTools (`bamCoverage`), samtools

---

## Input data

### Coordinate file (required for Steps 1–4)

A CSV with one row per intron. Required columns:

| Column | Description |
|--------|-------------|
| `chrom` | Chromosome (e.g. `chr1`) |
| `start` | Intron start (0-based) |
| `end` | Intron end |
| `strand` | `+` or `-` |
| `unique_ID` | Unique intron identifier |

Optional columns:

| Column | Description |
|--------|-------------|
| `expression` | Subset label (e.g. `high_retention` / `low_retention`) — used by Steps 2–3 to split data |

### BigWig files (required for Step 0)

Strand-unstranded RNA-seq BigWig files for each timecourse condition.
Set `fine_tuning.bw_dir` and `fine_tuning.bw_files` in your config.

---

## Reproducing the Manuscript Analyses

The exact YAML configuration files and processed input data used to generate
the figures are located in the `manuscript_runs/` directory.

```bash
# Reproduce the motif discovery analysis
python scripts/01_get_attributions.py --config manuscript_runs/fig_X.yaml
python scripts/02_run_modisco.py      --config manuscript_runs/fig_X.yaml
python scripts/03_run_enrichment.py   --config manuscript_runs/fig_X.yaml
python scripts/04_map_motifs.py       --config manuscript_runs/fig_X.yaml
```

The fine-tuned model checkpoint used in the manuscript is available on Zenodo:

```
DOI: https://doi.org/10.5281/zenodo.19296926
File: borzoi_mm10_delayed_splicing_finetuned.ckpt
```

Download it into a `weights/` directory before running the pipeline:

```bash
mkdir -p weights
wget -O weights/borzoi_mm10_delayed_splicing_finetuned.ckpt \
  "https://zenodo.org/records/19296926/files/borzoi_mm10_delayed_splicing_finetuned.ckpt"
```

---

## Usage

### 1. Create a config file

```bash
cp config/example_config.yaml config/my_run.yaml
# Edit paths and settings in my_run.yaml
```

The key fields to set:

```yaml
experiment_dir:  "/path/to/outputs/my_run"
coord_file_path: "/path/to/introns.csv"
gtf_file:        "/path/to/gencode.vM23.annotation.gtf"
species:         "mouse"
model_selection: "fine_tuned"
checkpoint_path: "weights/borzoi_mm10_delayed_splicing_finetuned.ckpt"
```

See [config/example_config.yaml](config/example_config.yaml) for all options
with inline documentation.

### 2. Run the pipeline

```bash
# Step 0: fine-tune Borzoi (~hours to days depending on accessible compute)
python scripts/00_fine_tune.py --config config/my_run.yaml

# Step 1: compute attributions (~minutes to hours depending on dataset size and GPU)
python scripts/01_get_attributions.py --config config/my_run.yaml

# Step 2: motif discovery
python scripts/02_run_modisco.py --config config/my_run.yaml

# Step 3: enrichment analysis
python scripts/03_run_enrichment.py --config config/my_run.yaml

# Step 4: generate plots
python scripts/04_map_motifs.py --config config/my_run.yaml
```

Steps must be run in order: each step reads outputs from the previous one.
Step 0 only needs to be run once; the checkpoint can be reused across multiple
attribution runs by setting `checkpoint_path` in each config.

---

## Output structure

```
experiment_dir/
├── attributions.pkl              # (N, 4, L) numpy array of attributions
├── input_seqs.pkl                # list of N input sequences (strings)
├── element_names_list.pkl        # list of N element names
├── attribution_mapping.csv       # coord_index → attribution_index mapping
│
├── all_seqs_modisco/
│   └── masked_50bp_flank/
│       ├── modisco_report.h5     # raw TF-MoDISco output
│       ├── forward.meme          # de novo motifs (forward strand)
│       └── combined.meme         # forward + reverse-complement motifs
│
├── fasta_files/
│   └── all_seqs.fa
│
├── enrichment/
│   └── all_seqs_modisco/
│       └── primary_seqs_vs_tnf_controls_seqs/
│           ├── sea/
│           │   ├── sea.tsv
│           │   ├── sea_enrichment.csv
│           │   └── sea_enrichment_scatter.png
│           └── fimo/
│               ├── fimo_enrichment.csv
│               └── fimo_enrichment_scatter.png
│
└── all_seqs_modisco/
    └── masked_50bp_flank/
        └── plots/
            ├── indexing_df.csv
            ├── elements_df.csv
            └── modiscolite/
                └── pos_pattern_0/
                    ├── pos_pattern_0_hits.csv
                    └── <intron_name>.png
```

---

## Configuration reference

All pipeline options are documented in [config/example_config.yaml](config/example_config.yaml).

**Step 0 key options** (under `fine_tuning:`)
- `bw_files` / `bw_dir`: BigWig RNA-seq inputs
- `trans_func`: label transform (`log1p` recommended)
- `dataset_cache_dir`: where to cache pre-built datasets
- `train_params.loss`: `mse` (used in manuscript)
- `train_params.max_epochs` / `lr`: training schedule

**Step 1 key options**
- `centering_mode`: `intron_only` (centre input on intron)
- `attr_respect_to`: `intron_only` (aggregate predictions over intron)
- `attribution_method`: `inputxgradient` (default)

**Step 2 key options**
- `mask_mode`: `intron_only` (mask signal outside intron ± flank)
- `flank`: list of flank sizes in bp
- `modisco_len`: MoDISco seqlet search window (bp)

**Step 3 key options**
- `enrichment.comparisons`: list of `{primary, control}` pairs

**Step 4 key options**
- `motif_mapping.modisco_window`: must match `modisco_len` from Step 2
- `motif_mapping.bigwig_dir`: optional read-density tracks overlay

---

## Dependencies

| Package | Role |
|---------|------|
| [gReLU](https://github.com/Genentech/gReLU) | Borzoi model loading, fine-tuning, attribution calculation |
| [tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite) | *De novo* motif discovery |
| [tangermeme](https://github.com/jmschrei/tangermeme) | FIMO motif scanning |
| [MEME Suite](https://meme-suite.org/) | SEA enrichment analysis |
| [logomaker](https://github.com/jbkinney/logomaker) | Sequence logo plots |
| [pyBigWig](https://github.com/deeptools/pyBigWig) | Read-density track overlays |
| [genomepy](https://github.com/vanheeringen-lab/genomepy) | Reference genome download and sequence fetching |
| [wandb](https://wandb.ai) | Fine-tuning experiment tracking |

---

## Citation

If you use this pipeline, please cite:

- **Borzoi**: Linder J, et al. *Predicting RNA-seq coverage from DNA sequence as a unifying model of gene regulation.* Nature Genetics (2025). https://doi.org/10.1038/s41588-024-02053-6
- **gReLU**: Nair S, et al. *gReLU: a Python library to train, interpret, and apply deep learning models to genomics.* (2024).
- **TF-MoDISco / modisco-lite**: Shrikumar A, et al. (2020); Trofimova D & Shrikumar A (2023).
- **MEME Suite**: Bailey TL, et al. *The MEME Suite.* Nucleic Acids Research (2015).

> **This analysis**: This analysis: Dearborn J.S., Frankiw L., Limoge D.W., Burns C.H., Vlach L., Turpin P., Kirch T., Miller Z.D., Dowell W., Languon S., Garcia‑Flores Y., Cockrell R.C., Baltimore D., Majumdar D. *Programmed Delayed Splicing: A Mechanism for Timed Inflammatory Gene Expression*. eLife (manuscript under review, 2026).

---

## License

[MIT](LICENSE)
