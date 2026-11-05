# Acceptance report — observation-supported UIE

Date: 2026-11-05. Statuses are **agent-proposed**, derived from artifacts by
`code/tools/make_acceptance_json.py`; no external verifier has been assigned.
Machine-readable version: [`acceptance_report.json`](<acceptance_report.json>).

| item | status | one-line reason |
|---|---|---|
| performance | `unverifiable` | no qualified frozen SOTA under the same protocol |
| novelty | `unsupported` | support does not separate from ordinary + ensemble uncertainty (H2) |
| mechanism | `unsupported` | the gate makes chroma worse; matched controls behave the same (H3) |
| reproducibility | `incomplete` | no three-seed training, no canonical splits, no sealed test |

`unsupported` means *the evidence in hand does not support the claim*. It does
not mean the direction is impossible.

## 1. Performance

Gate: `max(0,(S−R)/S) ≤ 0.05` on every PSNR (dB) and SSIM for UIEB-R90 / LSUI /
UFO-120. **Not evaluated**, because `R` is not established.

* 21 local-checkpoint reruns complete, zero errors, full test splits
  (`docs/sota_registry.csv`, `artifacts/baseline_*.json`).
* Five of seven methods reproduce the supplied `UIE-data.xlsx` figures within
  ±0.16 dB; the two that do not (CDF-UIE +1.71 dB, SFG-Net +2.49 dB on UIEB) are
  exactly the two whose published protocol is native-resolution or centre-cropped.
  So the reruns are credible — the *denominator* is the problem.
* MambaCue is a legacy saved-tile reference only: different UFO preprocessing
  (240×320), and Phaseformer beats it on LSUI (32.51 vs 31.20 dB) under the
  unified protocol. Its unified columns are blank and no gap is computed.
* Baseline checkpoint provenance is unverified (no git checkout, READMEs describe
  local domain adaptation, SFG-Net's README says weights were to be released
  later). Recorded as `local_checkpoint_rerun_provenance_unverified`.
* WWE-UIE and WaterNet: no main-network weights on the server → `not_run`, no
  estimated numbers.

## 2. H2 — incremental predictive value (decisive)

Pre-declared rule: support must lift R² over the ordinary baseline with a
bootstrap CI excluding 0 for chroma/relative targets, **and** beat its shuffled
and noise controls. 89 UIEB images × 3 backbones; 428 LSUI images × 1 backbone;
5 image-disjoint folds × 3 seeds; 800 image-clustered draws.

| domain | target | F0 | F1 +support | F2 shuffled | F3 noise | Δ(F1−F0) CI |
|---|---|---|---|---|---|---|
| UIEB | PSNR | −0.013 | +0.012 | −0.002 | +0.031 | [−0.050, +0.102] |
| UIEB | SSIM | 0.062 | 0.095 | 0.034 | 0.025 | [−0.018, +0.108] |
| UIEB | chroma | 0.106 | 0.098 | 0.071 | 0.107 | [−0.045, +0.043] |
| LSUI | PSNR | 0.173 | 0.094 | 0.064 | 0.073 | **[−0.133, −0.025]** |
| LSUI | SSIM | 0.186 | 0.120 | 0.101 | 0.081 | **[−0.104, −0.032]** |
| LSUI | chroma | 0.101 | 0.113 | 0.077 | 0.079 | [−0.044, +0.090] |

The two intervals that exclude zero are negative. Seven Gaussian noise columns
(F3) reach R² = 0.107 on UIEB chroma — the size of every effect in the table.
F1 does beat F2 in 6/6 cells (significantly in one), so the statistic is not
noise; it is **redundant** given the ordinary baseline, which is the precise
sense in which the novelty claim fails.

## 3. H3 — mechanism

Eleven frozen-output variants, 89 UIEB images, strength 1
(`artifacts/h3_mechanism_v2.json`; intervals `artifacts/h3_mechanism_v2_intervals.json`).

| variant | Δchroma % | ΔPSNR | ΔSSIM |
|---|---|---|---|
| **support_g** | **+1.70** | −0.069 | −0.0010 |
| const_g | +0.61 | −0.016 | −0.0006 |
| random_g | +0.62 | −0.017 | −0.0006 |
| degrade_g | +0.75 | −0.023 | −0.0007 |
| lowpass | +0.48 | −0.038 | −0.0038 |
| oracle_g | +0.25 | −0.012 | −0.0006 |
| project (self) | +0.00 | 0.000 | 0.0000 |
| project_oracle | −6.91 | +1.316 | +0.0009 |
| greyworld | +27.62 | −1.634 | −0.0123 |
| gain_oracle | +0.27 | +1.697 | +0.0004 |

Bootstrap: support_g − base = PSNR −0.0693 [−0.1604, +0.0028], SSIM −0.00104
[−0.00191, −0.00031], chroma +0.00219 [+0.00043, +0.00425] — the chroma
interval excludes zero on the **worse** side.

Support is not doing something the controls cannot do: three unrelated spatial
patterns (constant, random, degradation-matched) produce +0.61/+0.62/+0.75 %,
support produces +1.70 % of the same sign. And the reference-aware oracle
projection *loses* PSNR, which means the anchoring itself is wrong — stepping an
output along its own gauge family moves it away from the truth.

Invalidated: `artifacts/h3_mechanism.json` (4-image version). `apply_chroma` was
not the inverse of `chroma_opp`, so every chroma number in it is meaningless.
Retained, labelled `invalid`.

## 4. H1 — physical meaning

* Controlled detectability (416 rows, analytic scenes): AUC error 0.0055 mean /
  0.0630 max under the matched model, 0.0193 mean under response mismatch. The
  *implementation* tracks theory. No network is involved.
* Block-level, corrected code (`artifacts/h1_error_form_v2.json`): chroma error
  ρ(support) = +0.072 raw, **−0.000** partial given ordinary statistics, while
  forward statistics keep +0.196.
* The earlier +0.173 figure came from a gain bug in `direction_info` and is
  withdrawn.
* The "78 % of error is physically inconsistent" claim is withdrawn: a two-pixel
  counterexample (`dd = [+0.1, −0.1]`, identical render) leaves 99.69 % of the
  error outside the constant-step family, so family-external ≠ inconsistent.

## 5. Reproducibility

Have: protocol anchored to the MambaCue repo's own recorded result (≤0.04 dB);
SHA-256 manifest of 13 578 files and 6 789 pairs; environment freeze; per-experiment
ledger with commands and status (`docs/experiment_ledger.csv`); logs and artifacts
for every number in this report; instrument checks the imported functions, not a
copy.

Missing: three-seed training study; canonical split provenance (the supplied
UIEB 89 / LSUI 428 are not proven to be UIEB-800/90 or LSUI-3879/400); publication-
verified baseline weights; a sealed test set (the public test splits were
inspected during protocol work, so everything here is `exploratory`).

## 6. Claims withdrawn from earlier drafts

| withdrawn | why |
|---|---|
| universal "observation-consistency" failure explains most colour error | depends on a GT-anchored default model; pixel-varying depth refutes the inference |
| 78 % family-external error is physically inconsistent | two-pixel counterexample with identical render |
| support gate improves chroma by 11.6 % | `apply_chroma` inverse bug |
| misfit noise (`χ²/dof = 1`) is calibration | it is variance normalisation, not calibration |
| H2 first version rejects ordinary uncertainty | centred CV + held-out target means + gain bug; not a deployable comparison |
| MambaCue is the frozen SOTA | saved tiles, different UFO preprocessing, beaten on LSUI |

## 7. What would change the verdicts

1. Publication-verified weights for ≥2 methods → performance can leave `unverifiable`.
2. Support evaluated over ≥1000 images or with a power analysis → H2 could
   distinguish "no information" from "underpowered".
3. A backbone trained *with* the statistic → the only experiment that could
   rescue H3; never run here.
4. A never-inspected test split → any confirmatory claim.
