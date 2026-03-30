#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# Usage:
#   BAM input dir: arg1 (or BAM_INPUT_DIR env)
#   BW output dir: arg2 (or BW_OUTPUT_DIR env)
# Options (env or exported):
#   NORMALIZE (None|CPM|RPKM|BPM)   default: None
#   BIN_SIZE                        default: 10
#   THREADS                         default: 4
#   SKIP_DUPLICATES (0|1)           default: 0
#   STRANDED (auto|none|forward|reverse) default: auto (will prompt if auto and ambiguous)
#   EXTEND (int or "auto")          default: auto (estimate for single-end; no extension for paired-end)
#   BAM_PATTERN                      glob pattern for bams, default: *.bam
#
# Example:
#   NORMALIZE=CPM BIN_SIZE=1 THREADS=8 SKIP_DUPLICATES=1 STRANDED=none \
#     /path/to/bam_dir /path/to/bw_dir

BAM_INPUT_DIR="${1:-${BAM_INPUT_DIR:-}}"

if [[ -z "$BAM_INPUT_DIR" ]]; then
  echo "Usage: $0 <bam_input_dir> [bw_output_dir]"
  echo "       or set BAM_INPUT_DIR env variable"
  exit 1
fi

NORMALIZE="${NORMALIZE:-None}"
BIN_SIZE="${BIN_SIZE:-1}"
THREADS="${THREADS:-16}"
SKIP_DUPLICATES="${SKIP_DUPLICATES:-0}"
STRANDED="${STRANDED:-auto}"   # auto|none|forward|reverse
EXTEND="${EXTEND:-auto}"       # integer or "auto"
BAM_PATTERN="${BAM_PATTERN:-*.bam}"

_parent_dir="$(dirname "$BAM_INPUT_DIR")"
_norm_label="$(echo "${NORMALIZE:-None}" | tr '[:upper:]' '[:lower:]')"
if [[ "$_norm_label" == "none" || "$_norm_label" == "none" ]]; then
  _norm_label="no_norm"
fi

BW_OUTPUT_DIR="${2:-${BW_OUTPUT_DIR:-${_parent_dir}/bw/${_norm_label}}}"

mkdir -p "$BW_OUTPUT_DIR"

# tools check
for t in bamCoverage samtools; do
  if ! command -v "$t" &>/dev/null; then
    echo "Required tool '$t' not found in PATH. Please install deepTools (bamCoverage) and samtools."
    exit 1
  fi
done

# Helper: ensure BAM is indexed
ensure_index() {
  local bam="$1"
  if [[ ! -f "${bam}.bai" && ! -f "${bam%.bam}.bai" ]]; then
    echo "Index missing for $bam -> creating with samtools index"
    samtools index "$bam"
  fi
}

# Helper: detect if BAM is paired-end
is_paired() {
  local bam="$1"
  if samtools view -H "$bam" | grep -q "^@"; then
    local c
    c=$(samtools view -c -f 1 -F 12 "$bam" 2>/dev/null || echo 0)
    (( c > 0 )) && return 0 || return 1
  else
    return 1
  fi
}

# Estimate fragment length for single-end (numeric only)
estimate_fragment() {
  local bam="$1"
  # Average read length from first 50k alignments; fallback 200
  local len
  # Avoid pipefail killing the subshell when upstream gets SIGPIPE after awk exits early
  set +o pipefail
  len=$(samtools view -@ 1 "$bam" 2>/dev/null | awk '
    {
      l = length($10);
      if (l > 0) { sum += l; cnt++ }
      if (cnt >= 50000) { print int(sum/cnt); exit }
    }
    END {
      if (cnt > 0) print int(sum/cnt);
    }' 2>/dev/null || true)
  set -o pipefail

  # Fallback if empty/non-numeric
  if [[ -z "$len" || ! "$len" =~ ^[0-9]+$ ]]; then
    len=200
  fi
  echo "$len"
}

echo "Input BAM dir: $BAM_INPUT_DIR"
echo "Output BW dir: $BW_OUTPUT_DIR"
echo "Normalization: $NORMALIZE  binSize: $BIN_SIZE  threads: $THREADS  skipDuplicates: $SKIP_DUPLICATES  stranded: $STRANDED"

shopt -s nullglob
bams=( "$BAM_INPUT_DIR"/$BAM_PATTERN )
shopt -u nullglob

if [ ${#bams[@]} -eq 0 ]; then
  echo "No BAM files found in $BAM_INPUT_DIR with pattern $BAM_PATTERN"
  exit 0
fi

for bam in "${bams[@]}"; do
  base=$(basename "$bam" .bam)
  out="$BW_OUTPUT_DIR/${base}.bw"

  echo "----------------------------------------"
  echo "Processing $base"
  start=$SECONDS

  ensure_index "$bam"

  if is_paired "$bam"; then
    paired=1
  else
    paired=0
  fi

  # extend args
  extend_args=()
  if [[ "$EXTEND" == "auto" ]]; then
    if [[ $paired -eq 0 ]]; then
      frag=$(estimate_fragment "$bam")
      extend_args=( --extendReads "$frag" )
    fi
  elif [[ "$EXTEND" =~ ^[0-9]+$ ]]; then
    extend_args=( --extendReads "$EXTEND" )
  fi

  # strandedness
  read_strand_args=()
  if [[ "$STRANDED" == "auto" ]]; then
    echo "STRANDED=auto -> assuming unstranded. To force, set STRANDED=forward|reverse"
  elif [[ "$STRANDED" == "forward" ]]; then
    read_strand_args=( --filterRNAstrand forward )
  elif [[ "$STRANDED" == "reverse" ]]; then
    read_strand_args=( --filterRNAstrand reverse )
  fi

  # normalization (omit when None)
  _norm_up="$(echo "${NORMALIZE:-None}" | tr '[:lower:]' '[:upper:]')"
  norm_args=()
  if [[ -n "$_norm_up" && "$_norm_up" != "NONE" && "$_norm_up" != "NO_NORM" ]]; then
    norm_args=( --normalizeUsing "$NORMALIZE" )
  fi

  # duplicates
  dup_args=()
  if [[ "$SKIP_DUPLICATES" == "1" || "$SKIP_DUPLICATES" == "true" ]]; then
    dup_args=( --skipDuplicates )
  fi

  cmd=( bamCoverage -b "$bam" -o "$out" --binSize "$BIN_SIZE" -p "$THREADS" )
  cmd+=( "${dup_args[@]}" "${norm_args[@]}" "${extend_args[@]}" "${read_strand_args[@]}" )

  printf -v cmd_str '%q ' "${cmd[@]}"
  echo "Running: $cmd_str"
  "${cmd[@]}"

  duration=$((SECONDS - start))
  echo "Converted $bam -> $out (duration ${duration}s)"
done

echo "All BAM -> BigWig conversions finished."