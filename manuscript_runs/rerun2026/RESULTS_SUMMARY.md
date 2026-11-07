# Fig 6 motif re-run on poly(A)/internal-priming-filtered slow introns

Re-run of the motif-discovery pipeline (attributions -> modisco -> SEA/FIMO
enrichment) after removing introns flagged for internal priming (genomic A-run
>= 12) or an annotated intronic poly(A) site. Baseline (unfiltered) and filtered
sets were run through the **same pipeline** in the `grelu-test` env with the
Zenodo LoRA checkpoint (3-output model, 18/20/30-min heads).

Filtered sets: bottom-50 -> 31 kept; bottom-150 -> 120 kept.

## Headline result (intron-only motifs; Fig 6 main = bottom-50)

| set | n motifs | max A-content | # A>=0.80 | # GA>=0.80 |
|---|---|---|---|---|
| bottom-50 baseline  | 34 | 0.86 | **1** | 2 |
| bottom-50 filtered  | 23 | 0.67 | **0** | 2 |
| bottom-150 baseline | 42 | 0.83 | 1 (`AATAAA`) | 4 |
| bottom-150 filtered | 46 | 0.86 | 1 | 7 |

- **The clean poly-A / A-rich motif is eliminated by filtering in the bottom-50**
  (A=0.86 -> none >= 0.80). The baseline even rediscovers the canonical poly(A)
  signal `AATAAA` in the bottom-150 — direct evidence the slow set carries
  internal-priming / intronic-polyA signal. => the A-rich motif (old Fig 6C) is
  largely an internal-priming artifact, as the reviewer suspected.
- **The GA-rich primary motif is robust**: discovered de novo in every condition
  (2-7 motifs with GA>=0.80), with strong, specific enrichment
  (SEA: ~26-40% of slow introns vs 0-0.16% of TNF-control introns).
- A-rich motifs persist weakly in the larger bottom-150 (A-tracts are common in
  introns generally — consistent with the reviewer's "present in >50% of all
  introns" point), which is why the bottom-50 is the cleaner main set.

## Suggested manuscript/rebuttal framing
1. Keep the GA-rich motif as the robust, candidate finding (survives filtering).
2. Remove or explicitly demote the A-rich motif (old Fig 6C) as internal-priming/
   intronic-polyA-confounded; report the filtering + re-run as the evidence.
3. State the slow-intron subsets (bottom-50 / bottom-150, AUC-ranked) and the
   exclusion filter explicitly in Methods.

## Outputs
`results/rerun2026/io_{50,150}_{orig,filt}/` — each has:
- `all_seqs_modisco/masked_-5bp_flank/forward.meme` (discovered motifs)
- `enrichment/all_seqs_modisco/all_seqs_vs_tnf_controls_seqs/{sea,fimo}/` (enrichment + scatter PNGs)

## Known gaps / follow-ups
- **Supp Fig 8a/c (adjacent-exon masking)** failed: `elements_df is required for
  exon-based modes` — step2 needs an exon-coordinate table not auto-generated
  here. Fixable (build elements_df from the GTF) if those supplementary panels
  are to be regenerated; otherwise keep originals with the same filtering caveat.
- Step-4 gene-map/track plots failed non-fatally (font + a `KeyError: 'name'` in
  the plotting path); the enrichment results above are unaffected.
- Env: run in `grelu-test` (peft 0.17.1 + transformers 4.46.3); the repo's
  `elife-delay-splice` env has an incompatible transformers 5.6.2.
