#!/usr/bin/env bash
# Fig 6 / Supp Fig 8 re-run on internal-priming/poly(A)-filtered slow-intron sets.
# Computes attributions once per (set,variant), shares them with the adjacent-exon
# masking, then runs modisco -> enrichment -> map_motifs for both io and ae.
set -u
REPO=/mnt/data_sda/jdearborn/scratch/elife_delay_splice
PY=/mnt/data_sda/miniconda3/envs/grelu-test/bin/python3
export PATH=/home/jdearborn/meme/bin:$PATH
CFG=$REPO/manuscript_runs/rerun2026
RES=$REPO/results/rerun2026
LOG=$CFG/run_pipeline.log
cd "$REPO"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
run(){ log "RUN: $*"; "$@" >>"$LOG" 2>&1; local rc=$?; log "  -> exit $rc"; return $rc; }

log "=== pipeline start; sea=$(command -v sea) ==="
log "checkpoint: $(ls -la $REPO/weights/*.ckpt 2>/dev/null)"

# (set,variant) pairs; io attributions are shared into the ae experiment dir
for pair in 50_filt 50_orig 150_filt 150_orig; do
  io_dir=$RES/io_${pair}
  ae_dir=$RES/ae_${pair}
  log "########## SET ${pair} ##########"

  # --- step 1: attributions (io config) ---
  if [ -f "$io_dir/attributions.pkl" ]; then
    log "attributions exist for io_${pair}; skipping step1"
  else
    run $PY scripts/01_get_attributions.py --config $CFG/config_io_${pair}.yaml || { log "step1 FAILED for ${pair}; skipping set"; continue; }
  fi

  # --- share attributions into ae dir ---
  mkdir -p "$ae_dir"
  for f in attributions.pkl input_seqs.pkl element_names_list.pkl attribution_mapping.csv; do
    [ -f "$io_dir/$f" ] && cp -f "$io_dir/$f" "$ae_dir/$f"
  done
  log "copied attribution outputs io_${pair} -> ae_${pair}"

  # --- steps 2,3,4 for both masks ---
  for mk in io ae; do
    c=$CFG/config_${mk}_${pair}.yaml
    log "---- ${mk}_${pair}: modisco ----"
    run $PY scripts/02_run_modisco.py    --config $c || log "step2 FAILED ${mk}_${pair}"
    log "---- ${mk}_${pair}: enrichment ----"
    run $PY scripts/03_run_enrichment.py --config $c || log "step3 FAILED ${mk}_${pair}"
    log "---- ${mk}_${pair}: map_motifs ----"
    run $PY scripts/04_map_motifs.py     --config $c || log "step4 FAILED ${mk}_${pair} (non-fatal)"
  done
done

log "=== PIPELINE COMPLETE ==="
