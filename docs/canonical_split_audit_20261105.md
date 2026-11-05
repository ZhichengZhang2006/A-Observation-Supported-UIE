# Canonical split audit (2026-11-05)

Scope: are the server directories a canonical, leak-free version of the benchmarks
this report names (UIEB-R90, LSUI, UFO-120)? Answer given below from hashes, not
from file counts. Everything here is derived from
`artifacts/provenance/{source_images.csv,split_pairs.csv,pairing_and_duplicates.json}`,
produced by `code/tools/inventory_research.py` (13 578 source files, 6 789 pairs,
manifest SHA-256 `14bc34127319d4e6f26fa6f3c69014a4c9af88dffb71b8b0f362d2825c908e6d`).

## 1. What is on the server

| dataset | train | val | test | pairing |
| --- | --- | --- | --- | --- |
| UIEB | 712 | 89 | 89 | 712/89/89 inputs and refs, 0 unmatched everywhere |
| LSUI | 3423 | 428 | 428 | 3423/428/428, 0 unmatched |
| UFO | 1200 | 300 | 120 | 1200/300/120, 0 unmatched |

`nonbijective_refs` is empty for all nine split directories: each input resolves to
exactly one reference and each reference to exactly one input, so there is no
ambiguous pairing inside a split.

## 2. Is it canonical? Partly — and the difference matters

UIEB here is **712/89/89 = 890 pairs**, the full UIEB size. The paper-facing name
"UIEB-R90" refers to a 90-image reference test set. This tree's `UIEB/test` has 89
files, not 90, and nothing in the tree states which reference list it came from.
The same holds for LSUI: 3423/428/428 sums to 4279, not the 3879/400 figures quoted
for the public release. **Counts are consistent within themselves but do not match
the canonical published splits**, and no provenance file in the tree resolves the
difference. This report therefore does not call these directories canonical, and
`docs/sota_registry.csv` reports them under the dataset names without asserting
canonical identity.

Consequence for claims: a number measured here is comparable to another number
measured here, and to the frozen-protocol anchor reproduced from MambaCue's own
saved outputs (24.4762 dB vs 24.4629 dB recorded). It is not automatically
comparable to a table in another paper that used a different 89/90 or 3879/400
selection.

## 3. Cross-split duplicate audit (exact bytes)

| dataset | split pair | identical inputs | identical refs | identical paired samples |
| --- | --- | --- | --- | --- |
| UIEB | train–val | 0 | 0 | 0 |
| UIEB | train–test | 0 | 0 | 0 |
| UIEB | val–test | 0 | 0 | 0 |
| LSUI | train–val | 12 | 14 | 10 |
| LSUI | train–test | 10 | 17 | 9 |
| LSUI | val–test | 2 | 5 | 2 |
| UFO | train–val | 1 | 1 | 1 |
| UFO | train–test | 3 | 3 | 3 |
| UFO | val–test | 2 | 2 | 2 |

UIEB is clean at the byte level across all three splits. LSUI and UFO are not: a
handful of held-out images are byte-identical to training images.

## 4. What the LSUI/UFO overlap does and does not prove

**Does not prove:** that any specific checkpoint consumed those files, that its
training pipeline sampled them, or that the affected test rows are biased by an
equal amount. Checkpoint provenance is itself unverified (see §5), so the
consumption question cannot be answered from the filesystem.

**Does prove:** the held-out sets are not disjoint from the training sets at the
byte level, so any per-image statistic computed on LSUI/UFO test carries a small
contamination risk that must be disclosed rather than assumed away. The affected
counts are small (9–10 paired LSUI samples out of 428, 3 of 120 UFO), which bounds
the aggregate effect but does not eliminate it.

Numbers in this report that use LSUI/UFO therefore carry the caveat "held-out
split contains N byte-identical training samples", and the H2 study (§ in
`docs/acceptance_report.md`) restricts its *fitting* pool to a val split whose
overlap is disclosed rather than claiming a clean split.

Near-duplicate detection (perceptual hashing, crops, re-encodes) was **not** run
to completion; earlier sampling found near-duplicates in LSUI with dHash. No
numeric bound on perceptual overlap is claimed here.

## 5. Weights

`artifacts/provenance/checkpoints.json` records path, SHA-256 and byte size for
every checkpoint used, and flags all of them `origin: unverified`. Reasons, from
the read-only source audit in `docs/literature_verification_20261105.md`:

- none of the nine model directories under `/home/user` is a git checkout
  (`git rev-parse` fails in all of them), so the shipped code cannot be tied to a
  release tag;
- several READMEs describe user-side adaptation and per-dataset retraining, and
  the SFG-Net README says pretrained weights would be uploaded later while the
  directory already contains many per-epoch files;
- MambaCue's README describes `variant_07` as a user-modified model.

So `local_checkpoint_rerun_provenance_unverified` is the strongest label the
evidence supports, and it is the label used in `docs/sota_registry.csv`.

## 6. References are not water-free radiance

Every `GT` directory contains human-restored references, not measurements of the
water-free scene. This is why the report never calls the reference "ground truth"
in a physical sense, and why the gauge/ambiguity results are stated as properties
of a *model plus a reference*, not of the scene.

## 7. Open items this audit could not close

1. Which reference list produced 89 UIEB test images (R90 minus one?) — unresolved.
2. Whether LSUI's 428/428 come from the public 3879/400 publication split — unresolved.
3. Perceptual/near-duplicate overlap — sampled, not exhaustively quantified.
4. Whether any checkpoint consumed the overlapping LSUI/UFO files — unresolvable
   from the filesystem; declared as a limitation.
