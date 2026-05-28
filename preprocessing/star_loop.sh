#!/bin/bash

# star alignment script adapted for single end reads here.

# Set the paths to STAR and the reference genome
# Override by setting STAR_PATH in your environment: export STAR_PATH=/path/to/STAR
STAR="${STAR_PATH:-$(which STAR 2>/dev/null || echo /usr/bin/STAR)}"
MAX_LOAD=16.0  # Maximum system load allowed to start the script

# Check if STAR exists
if [ ! -x "$STAR" ]; then
  echo "Error: STAR executable not found at $STAR. Set STAR_PATH=/path/to/STAR or add STAR to your PATH."
  exit 1
fi

# Prompt the user for the species associated with the data
read -p "Please enter the species (mouse or human): " SPECIES

# Set Reference genome based on species input
if [ "$SPECIES" = "mouse" ]; then
    read -p "Please enter the path to the STAR index directory for mouse: " REF_GENOME
elif [ "$SPECIES" = "human" ]; then
    read -p "Please enter the path to the STAR index directory for human: " REF_GENOME
else
    echo "Error: No reference genome set. Please specify 'mouse' or 'human'."
    exit 1
fi

if [ ! -d "$REF_GENOME" ]; then
    echo "Error: STAR index directory not found at '$REF_GENOME'."
    exit 1
fi

echo "Reference Genome Path: $REF_GENOME"

# get the path to the directory containing the fastq files
read -p "Please enter the path to the directory containing the fastq files: " INPUT_DIR

# Set output directories
PARENT_DIR=$(dirname "$INPUT_DIR")
echo "Parent directory: $PARENT_DIR"
OUTPUT_DIR="$PARENT_DIR/bam"        # Output BAM files
echo "Output directory: $OUTPUT_DIR"
STAR_LOG_DIR="$PARENT_DIR/star_logs" # STAR log output files
echo "STAR log directory: $STAR_LOG_DIR"

# Create the output and STAR logs directories if they don't exist
mkdir -p "$OUTPUT_DIR" "$STAR_LOG_DIR"

# Function to get the current system load (1-minute average)
get_current_load() {
  awk '{print $1}' /proc/loadavg
}

# Choose threads based on current load (set once per run)
current_load=$(get_current_load)

# bump this up to 16 or 24 if you have a high-load system
if (( $(echo "$current_load < $MAX_LOAD" | bc -l) )); then
  echo "System load ($current_load) is under the threshold ($MAX_LOAD). Proceeding with 8 threads."
  threadcount=8
else
  echo "WARNING: System load ($current_load) exceeds MAX_LOAD=$MAX_LOAD. Proceeding with reduced threads."
  threadcount=4
fi

# --- Paired-end-aware loop --------------------------------------------------
# Look for read1 files only to avoid processing mates twice.
shopt -s nullglob
r1_candidates=( "$INPUT_DIR"/*_1.fastq "$INPUT_DIR"/*_1.fastq.gz "$INPUT_DIR"/*_1.fq "$INPUT_DIR"/*_1.fq.gz \
                "$INPUT_DIR"/*_R1.fastq "$INPUT_DIR"/*_R1.fastq.gz "$INPUT_DIR"/*_R1.fq "$INPUT_DIR"/*_R1.fq.gz )
shopt -u nullglob

# If no R1-style names are present, fall back to treating any FASTQ as single-end
FORCE_SINGLE=0
if [ ${#r1_candidates[@]} -eq 0 ]; then
  shopt -s nullglob
  all_fastqs=( "$INPUT_DIR"/*.fastq "$INPUT_DIR"/*.fastq.gz "$INPUT_DIR"/*.fq "$INPUT_DIR"/*.fq.gz )
  shopt -u nullglob
  if [ ${#all_fastqs[@]} -eq 0 ]; then
    echo "No FASTQ files found in $INPUT_DIR"
    exit 0
  fi
  echo "No R1-style files found; treating all FASTQ as single-end"
  r1_candidates=( "${all_fastqs[@]}" )
  FORCE_SINGLE=1
fi

for r1 in "${r1_candidates[@]}"; do
  [[ -e "$r1" ]] || continue

  filename="$(basename "$r1")"

  if [[ $FORCE_SINGLE -eq 1 ]]; then
    # single-end sample base: strip common extensions
    sample="${filename%%.fastq.gz}"
    sample="${sample%%.fq.gz}"
    sample="${sample%%.fastq}"
    sample="${sample%%.fq}"
    mate=""
  else
    # derive sample base by stripping the _1/_R1 suffix and any following extension
    sample="${filename%_1*}"
    sample="${sample%_R1*}"

    # candidate mate names (try several replacement patterns)
    mate_candidates=(
      "${r1/_1./_2.}"
      "${r1/_1/_2}"
      "${r1/_R1./_R2.}"
      "${r1/_R1/_R2}"
    )

    mate=""
    for c in "${mate_candidates[@]}"; do
      [[ -f "$c" ]] && { mate="$c"; break; }
    done
  fi

  out_prefix="$OUTPUT_DIR/${sample}"
  output_bam="$OUTPUT_DIR/${sample}.bam"

  echo "Processing sample: $sample"
  echo "  R1: $r1"
  if [[ -n "$mate" ]]; then
    echo "  R2: $mate"
  else
    echo "  R2: (not found) - falling back to single-end"
  fi

  if [[ -f "$output_bam" ]]; then
    echo "  Output BAM exists, skipping."
    continue
  fi

  # determine if either file is gzipped (use extension check to avoid calling file() for many files)
  if [[ "$r1" == *.gz || "$mate" == *.gz ]]; then
    read_cmd="--readFilesCommand zcat"
  else
    read_cmd=""
  fi

  # build STAR command
  if [[ -n "$mate" ]]; then
    if [[ -n "$read_cmd" ]]; then
      $STAR --runThreadN "$threadcount" --genomeDir "$REF_GENOME" --readFilesIn "$r1" "$mate" $read_cmd --outFileNamePrefix "${out_prefix}" --outSAMtype BAM SortedByCoordinate
    else
      $STAR --runThreadN "$threadcount" --genomeDir "$REF_GENOME" --readFilesIn "$r1" "$mate" --outFileNamePrefix "${out_prefix}" --outSAMtype BAM SortedByCoordinate
    fi
  else
    # single-end fallback
    if [[ -n "$read_cmd" ]]; then
      $STAR --runThreadN "$threadcount" --genomeDir "$REF_GENOME" --readFilesIn "$r1" $read_cmd --outFileNamePrefix "${out_prefix}" --outSAMtype BAM SortedByCoordinate
    else
      $STAR --runThreadN "$threadcount" --genomeDir "$REF_GENOME" --readFilesIn "$r1" --outFileNamePrefix "${out_prefix}" --outSAMtype BAM SortedByCoordinate
    fi
  fi

  if [ $? -ne 0 ]; then
    echo "Error: STAR alignment failed for sample $sample"
    continue
  fi

  # move STAR output BAM to standardized filename (STAR uses <prefix>Aligned.sortedByCoord.out.bam)
  if [ -f "${out_prefix}Aligned.sortedByCoord.out.bam" ]; then
    mv "${out_prefix}Aligned.sortedByCoord.out.bam" "$output_bam"
  else
    echo "Warning: expected STAR BAM not found for $sample"
  fi

  # move STAR logs
  mv "${out_prefix}"*.out "${out_prefix}"*.tab "$STAR_LOG_DIR" 2>/dev/null || true
done
# ---------------------------------------------------------------------------

echo "Done."