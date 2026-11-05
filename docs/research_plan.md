# Research plan — observation-supported UIE

Status: retrospective design record. This document was assembled from the run
history on 2026-11-05; it is **not** a pre-registration and must not be cited as
one. Where a decision changed after a defect was found, the change is logged at
the end rather than silently rewritten.

## 1. Question

Given one underwater photograph `I` and a restoration `Ĵ` produced by any
enhancer, can we compute — from the photograph alone — *how much of `Ĵ`'s colour
and detail is actually pinned down by the observation*, and does that quantity
predict the enhancer's own error better than the ordinary uncertainty signals a
practitioner already has?

Two measurement objects were carried through the study:

* **Support (Fisher/Schur).** With a water model `I = g[b + (J - b)·exp(-βd)]`
  and nuisance parameters `ψ = (β, d, b, g)`, profile the likelihood and take the
  curvature along a candidate colour/detail direction. The profiled information
  `q_v` says how many σ of observation stand behind that direction. Algebraic
  novelty is explicitly disclaimed (Sampson & Melchior 2023; Clarkson 2019;
  Bakeer 2026 all handle nuisance-profiled Fisher information).
* **Ambiguity interval.** The width of the set of scene states that render `I`
  within the noise. In the exact gauge family `J' = b + (J-b)·exp(β·δd)`,
  `d' = d + δd` leaves the observation bit-identical.

## 2. Hypotheses and how each was to be decided

| id | hypothesis | decisive test | threshold fixed in advance |
|---|---|---|---|
| H1 | support is an independent physical quantity, not a re-description of brightness/contrast | controlled synthetic scenes with known and unknown water; detectability must follow the analytic prediction | AUC error vs `Φ(A√q/√2)` reported per condition |
| H2 | support adds prediction over ordinary uncertainty | learned heteroscedastic error model, F0 = ordinary + ensemble disagreement, F1 = F0 + support, F2 = F0 + shuffled support, F3 = F0 + noise columns | ΔPSNR ≥ 0.1 dB or ΔSSIM ≥ 0.001 to claim separability; CI must exclude 0 |
| H3 | the support statistic, used as a gate, improves the restoration it describes | 11 frozen-output variants on the same predictions, matched strength/receptive field/output quality | any improvement must beat `const_g`/`random_g`/`degrade_g` controls, not just `base` |
| — | performance | every method on UIEB-R90 / LSUI / UFO-120 within 5 % of frozen SOTA | `max(0,(S-R)/S) ≤ 0.05`, PSNR in dB |

## 3. Protocol (frozen before any number was produced)

* Whole image → 256×256 with PIL `BILINEAR`; **no crop, no pad**.
* PSNR: `skimage.metrics.peak_signal_noise_ratio(data_range=1.0)`.
* SSIM: `structural_similarity(channel_axis=-1, win_size=5, data_range=1.0)`;
  per-image values then averaged.
* The protocol was accepted only after it reproduced the MambaCue repo's own
  `result.txt` to within 0.04 dB on all three datasets from that repo's saved
  prediction tiles (E01).

## 4. Data

Server copy at `/home/user/UnderWaterDataset`: UIEB 712/89/89, LSUI
3423/428/428, UFO 1200/300/120 (train/val/test). These are **not** proven to be
the canonical UIEB-800/90 or LSUI-3879/400 splits; the counts differ and no
provenance document accompanies them. Full SHA-256 inventory, cross-split exact
duplicate counts and pairing manifests are in `docs/data_manifest.md` and
`artifacts/provenance/`.

Consequence recorded before modelling: because the public test splits were
already used to inspect method behaviour, no later experiment on them can be
called a sealed test. Every H2/H3 number is therefore labelled exploratory.

## 5. Work packages

| wp | content | artifact |
|---|---|---|
| WP1 | protocol verification + 21 local-checkpoint reruns | `artifacts/baseline_*_*.json`, `docs/sota_registry.csv` |
| WP2 | instrument implementation checks | `artifacts/instrument_checks_v3_*.json`, `code/obsui/{support,gauge,controlled}.py` |
| WP3 | H1 controlled detectability | `artifacts/controlled_detection_v2/` |
| WP4 | H2 incremental prediction (v3) | `artifacts/h2_v3*.json`, `code/tools/h2_risk_v3.py` |
| WP5 | H3 mechanism + matched controls | `artifacts/h3_mechanism_v2*.json`, `code/tools/mechanism.py` |
| WP6 | provenance, budget, packaging | `docs/*`, this repository |

## 6. Known gaps this plan does not close

* Two backbones are used for H2, but neither is retrained; the study cannot
  separate "support is uninformative" from "these two frozen models happen not
  to expose it".
* No three-seed training study, so H3 is a single frozen-output test.
* MC-dropout and deep-ensemble uncertainty are replaced by cross-method
  disagreement of seven frozen checkpoints. That is a weaker epistemic signal
  than a trained ensemble and is stated as such.
* WWE-UIE and WaterNet have no weights on the server and are reported as
  not-run rather than estimated.

## 7. Revision log

1. **2026-11-05 — `apply_chroma` inverse fixed.** The chroma encode/decode pair
   was not an exact inverse (missing `EPS` offset), invalidating the first H3
   run. Fixed; H3 re-run as v2. The old artifact is retained as invalid.
2. **2026-11-05 — `direction_info` gain fix.** Global direction features omitted
   the channel gain, so `q` was wrong by `gain²`. Fixed; H1/H2 features must be
   recomputed (the per-pixel H3 path was unaffected).
3. **2026-11-05 — basis rank fix.** `components(h, w, k)` returned `k` columns
   from a rank-`k-1` centred basis, silently adding an arbitrary null direction.
   Now SVD-truncated to the actual rank.
4. **2026-11-05 — `family_spread` argmin/argmax swap.** The "free PSNR" statistic
   was identically zero. Fixed and a zero step forced onto the grid.
5. **2026-11-05 — H2 redesigned.** The first H2 used centred-within-image CV and
   held-out target means; that scores relative position, not deployable absolute
   risk, and it counted five global features as free predictors. Replaced by v3
   (uncentred, train-only scaling, mirrored image-disjoint folds).
6. **2026-11-05 — train split abandoned as a training source.** The frozen UIEB
   checkpoints score ≈42.5 dB on their own train split (memorisation) versus
   24.5 dB on test, so the train split cannot stand in for deployment.
7. **2026-11-05 — MambaCue demoted from SOTA denominator.** Its numbers come from
   saved tiles under a different preprocessing on UFO, and Phaseformer beats it
   on LSUI under the unified protocol, so the registry reports it as a legacy
   reference only.
