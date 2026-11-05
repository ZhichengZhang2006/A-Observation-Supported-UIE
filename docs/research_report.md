# Observation-supported underwater image enhancement — research report

Date: 2026-11-05. Repository: `A-Observation-Supported-UIE`.
Mandate: `AutoResearch_A_Observation_Supported_UIE.md`.

**One line up front.** The measurement instrument was built and checked; the
strong version of the claim did not survive its own controls on the data
available here. Support does not demonstrably beat ordinary uncertainty +
ensemble disagreement at predicting restoration error (H2), and using it as a
gate makes colour worse, not better (H3). The performance gate is not settled
because the reference SOTA is not established. Everything below distinguishes
what was measured from what was hoped.

---

## 1. What was asked

Six quality items, each with an explicit pass rule: performance within 5 % of a
frozen SOTA on UIEB-R90 / LSUI / UFO-120 (PSNR in dB, relative), H1 (independent
physical evidence that the support statistic is not a restatement of the
observable), H2 (the statistic adds predictive power over ordinary uncertainty),
H3 (a mechanism showing it changes outcomes under matched controls),
reproducibility (manifests, hashes, environment, commands, logs, ledger, budget),
and honesty (unfavourable results retained).

## 2. Protocol, and why you should believe it

Frozen before any number: whole image → 256×256 PIL `BILINEAR`, no crop, no pad;
PSNR `skimage` `data_range=1.0`; SSIM `win_size=5`, `channel_axis=-1`,
`data_range=1.0`, per-image then mean.

The protocol was not accepted on assertion. The MambaCue repository saves its
predictions as `hstack(raw, GT, pred)` tiles; re-scoring its own tiles under the
frozen protocol reproduces the run's recorded `result.txt` to within 0.04 dB on
all three datasets:

| dataset | measured (frozen protocol) | recorded by the repo | Δ |
|---|---|---|---|
| UIEB | 24.4762 / 0.9221 | 24.4629 / 0.9224 | +0.013 dB |
| LSUI | 31.1957 / 0.9351 | — | — |
| UFO-120 | 28.5965 / 0.8780 | — | — |

That agreement is the only reason the protocol is treated as the authors' own
rather than an approximation of it. Note the second consequence: the frozen
protocol *is* the MambaCue protocol, which means MambaCue cannot be claimed to
be disadvantaged by it.

## 3. Performance: the reference SOTA is not settled

21 local-checkpoint reruns were completed (7 methods × 3 datasets, full test
splits, zero errors) — `docs/sota_registry.csv`, `artifacts/baseline_*.json`.

| method | UIEB PSNR/SSIM | LSUI PSNR/SSIM | UFO PSNR/SSIM |
|---|---|---|---|
| Phaseformer | 24.2302 / 0.9149 | **32.5089 / 0.9408** | **30.1545 / 0.8975** |
| AquaClarity | 23.5541 / 0.9145 | 30.1160 / 0.9404 | 28.3574 / 0.8819 |
| SS-UIE | 21.4540 / 0.8595 | 27.2907 / 0.9042 | 27.0979 / 0.8563 |
| FDCE-Net | 22.7114 / 0.8985 | 26.2825 / 0.9130 | 24.4445 / 0.8422 |
| CCL-Net | 21.1382 / 0.8619 | 25.9133 / 0.8794 | 24.7848 / 0.8012 |
| SFG-Net | 20.8171 / 0.8743 | 25.0092 / 0.9031 | 26.8374 / 0.8595 |
| CDF-UIE | 21.8999 / 0.8691 | 24.6539 / 0.8797 | 24.8719 / 0.7725 |
| MambaCue (saved tiles) | 24.4762 / 0.9221 | 31.1957 / 0.9351 | 28.5965 / 0.8780 |

Three things follow, and only the first is comfortable:

1. **Protocol sanity.** Five of seven methods land within ±0.16 dB of the
   figures supplied in `UIE-data.xlsx`. The two that do not (+1.71 dB CDF-UIE,
   +2.49 dB SFG-Net on UIEB) are exactly the two whose published evaluation
   protocol differs (native resolution / centre crop), so the gap is explained
   by protocol, not by a broken rerun.
2. **MambaCue is not a denominator.** Its numbers come from saved tiles under a
   different UFO preprocessing (240×320 packed tiles, no unified resize), and
   Phaseformer beats it by 1.3 dB on LSUI under the unified protocol. The
   registry therefore reports MambaCue as a *legacy reference*, with the unified
   columns blank and no 5 % gap computed against it.
3. **The 5 % gate cannot be declared passed.** Because the denominator is not
   established, `performance` is `unverifiable` here. This is a statement about
   missing evidence, not about the methods.

Checkpoint provenance is also unverified: none of the seven baseline directories
is a git checkout, several READMEs describe local domain adaptation, and the
SFG-Net README says pretrained weights were to be uploaded later while the
server copy has many per-epoch files. "Official weights" is therefore recorded
as `local_checkpoint_rerun_provenance_unverified` throughout — the numbers are
real measurements of *these files*, not certified reproductions of the papers.

## 4. H1 — is the statistic physically meaningful?

Two instruments, both model-conditional, both honest about it.

**(a) Controlled detectability** (`artifacts/controlled_detection_v2/`, 8 analytic
16×16 scenes, 13 conditions, 256 independent Gaussian draws per class = 416
rows). Predicted AUC from the profiled information `q` versus measured
Mann–Whitney AUC:

| regime | mean abs AUC error | max abs AUC error |
|---|---|---|
| matched model (linear) | 0.0055 | 0.0630 |
| response mismatch (model wrong) | 0.0193 | — |

The instrument tracks detectability when the model is right and degrades
predictably when it is wrong. This validates the *implementation* of the
statistic. It does **not** show that real enhancer error is what the statistic
predicts — no network was involved.

**(b) Block-level error form.** 89 UIEB images, 22 784 blocks, corrected support
code (`artifacts/h1_error_form_v2.json`):

| error form | ρ(support) | ρ(depth τ) | partial ρ support (controls) | partial ρ forward stats |
|---|---|---|---|---|
| abs | +0.014 | −0.046 | +0.071 | +0.112 |
| rel | +0.076 | +0.099 | +0.017 | +0.147 |
| **chroma** | **+0.072** | **+0.134** | **−0.000** | **+0.196** |
| luma | +0.078 | −0.021 | +0.118 | +0.089 |

For the target this project cares about — chromatic error — the support
statistic's raw association is weak and **vanishes entirely** once ordinary
image statistics are controlled (−0.000), while the plain forward statistics
retain +0.196. The chroma-over-luma error ratio has ρ(support) = +0.003.

The first version of this artifact reported ρ = +0.173 for chroma; that number
was produced before the `direction_info` gain bug was fixed and is retained as
superseded. The correction is not cosmetic: it removes the whole effect.

**Verdict: `partial` for the instrument, `unsupported` for the strong reading.**
The instrument tracks detectability where the truth is known (a); on real data
it does not carry independent information about colour error (b).

## 5. H2 — does support add anything over ordinary uncertainty?

Design (declared in `docs/research_plan.md` §2 and `code/tools/h2_risk_v3.py`):
per-image heteroscedastic Gaussian head (mean + log-variance, NLL, early stop on
an image-disjoint tenth), image-disjoint 5-fold CV, out-of-fold predictions
pooled, 3 seeds, image-clustered bootstrap (1000 draws). Four equal-capacity
arms:

* **F0 ordinary** — 9 observation statistics **including fitted optical depth**
  (deliberately included: H1 had shown that not controlling depth makes the
  comparison a straw man) + 4 ensemble-disagreement statistics.
* **F1 + support** — F0 plus 7 support statistics.
* **F2 + shuffled support** — the same 7 columns permuted across images.
* **F3 + noise** — F0 plus 7 Gaussian columns.

Results are in §5.1 below once the run completes; the pass rule was fixed in
advance: support must raise R² for the chroma/relative targets with a bootstrap
CI excluding 0, and must beat F2/F3, not just F0.

### 5.1 Results

UIEB: 89 images × 3 backbones = 267 rows. LSUI: 428 images × 1 backbone = 428
rows (the other two LSUI backbones were cut for compute; this is a coverage gap,
not a zero). 5 folds × 3 seeds × 250 epochs, 800 image-clustered bootstrap draws.

| domain | target | F0 ordinary | F1 +support | F2 shuffled | F3 noise | ΔR²(F1−F0) [95 % CI] | ΔR²(F1−F2) [95 % CI] |
|---|---|---|---|---|---|---|---|
| UIEB | PSNR | −0.013 | +0.012 | −0.002 | +0.031 | +0.024 [−0.050, +0.102] | +0.012 [−0.060, +0.082] |
| UIEB | SSIM | 0.062 | 0.095 | 0.034 | 0.025 | +0.037 [−0.018, +0.108] | +0.063 [+0.000, +0.127] |
| UIEB | chroma | 0.106 | 0.098 | 0.071 | 0.107 | −0.007 [−0.045, +0.043] | +0.028 [−0.007, +0.068] |
| LSUI | PSNR | 0.173 | 0.094 | 0.064 | 0.073 | **−0.080 [−0.133, −0.025]** | +0.029 [−0.027, +0.083] |
| LSUI | SSIM | 0.186 | 0.120 | 0.101 | 0.081 | **−0.067 [−0.104, −0.032]** | +0.017 [−0.050, +0.078] |
| LSUI | chroma | 0.101 | 0.113 | 0.077 | 0.079 | +0.015 [−0.044, +0.090] | +0.040 [−0.011, +0.100] |

Three readings, in order of importance:

1. **The pre-declared rule fails.** Support must lift R² over F0 with a CI
   excluding 0 on the chroma/relative targets. It does not: the two intervals
   that exclude 0 (LSUI PSNR and SSIM) are **negative** — adding the support
   columns makes the risk model worse.
2. **The noise floor is the size of the effect.** F3, seven pure Gaussian
   columns of the same shape, reaches R² = 0.031 and 0.107 on UIEB — as high as
   F1 on PSNR and higher than F2 on chroma. At n = 89 images, ΔR² of ±0.03
   cannot be distinguished from feature noise, which is exactly the audit's
   "ΔPSNR ≤ 0.1 dB" falsifier in R² clothing.
3. **The support columns are not pure noise, but they are redundant.** F1 beats
   F2 (its own permutation) in all six cells, and once significantly (UIEB SSIM,
   +0.063 [+0.000, +0.127]). So the statistic carries *some* real information
   about error — the problem is that F0 already contains it. That is the
   difference between "the measurement is meaningless" and "the measurement is
   not new", and only the second is supported here.

Spearman correlations between predicted and true risk stay around 0.14–0.50 with
F0 already at 0.14–0.58, so ordering, not just R², shows no separation.

**Verdict: `unsupported`.** With the depth-controlled ordinary baseline plus
seven-model ensemble disagreement, support adds nothing and on LSUI costs
0.07–0.08 R². The one thing it would take to overturn this — a backbone trained
*with* the statistic — was never run.

## 6. H3 — does the mechanism change outcomes?

Eleven frozen-output variants on the same 89 UIEB predictions, strength 1,
identical outputs except for the applied correction
(`artifacts/h3_mechanism_v2.json`, intervals in
`artifacts/h3_mechanism_v2_intervals.json`).

| variant | chroma | Δchroma % | PSNR | ΔPSNR | SSIM |
|---|---|---|---|---|---|
| base | 0.1288 | — | 24.476 | — | 0.9221 |
| **support_g** | 0.1310 | **+1.70** | 24.407 | −0.069 | 0.9211 |
| const_g | 0.1296 | +0.61 | 24.460 | −0.016 | 0.9215 |
| random_g | 0.1296 | +0.62 | 24.459 | −0.017 | 0.9215 |
| degrade_g | 0.1298 | +0.75 | 24.453 | −0.023 | 0.9214 |
| lowpass | 0.1294 | +0.48 | 24.438 | −0.038 | 0.9183 |
| oracle_g | 0.1291 | +0.25 | 24.464 | −0.012 | 0.9215 |
| project (self) | 0.1288 | +0.00 | 24.476 | +0.000 | 0.9221 |
| project_oracle | 0.1199 | −6.91 | 25.792 | +1.316 | 0.9230 |
| greyworld | 0.1644 | +27.62 | 22.842 | −1.634 | 0.9098 |
| gain_oracle | 0.1291 | +0.27 | 26.173 | +1.697 | 0.9225 |

Bootstrap (2000 image draws): support_g vs base gives PSNR −0.0693
CI [−0.1604, +0.0028], SSIM −0.00104 CI [−0.00191, −0.00031], chroma +0.00219
CI [+0.00043, +0.00425]. The chroma interval **excludes zero on the bad side**:
gating by support makes chroma measurably worse.

And the crucial control comparison: `const_g`, `random_g` and `degrade_g` all
improve *less* than `support_g` distorts — i.e. the four "different" spatial
patterns produce chroma deltas of +1.70 / +0.61 / +0.62 / +0.75 %, so support is
not doing something its controls cannot do; it is doing the same thing slightly
harder. The oracle variant (choosing the step with reference knowledge) also
loses PSNR, which means the anchoring itself is wrong: stepping the output along
its own gauge family moves it away from the truth.

**Verdict: `unsupported` for the mechanism as stated.** The earlier, widely
quoted "support gate improves chroma by 11.6 %" number came from a code path
where `apply_chroma` was not the inverse of `chroma_opp` (an `EPS` offset was
omitted from the decode), and must not be cited; that artifact is retained as
`artifacts/h3_mechanism.json` with status `invalid`.

## 7. The gauge result — what is actually established

The exact symmetry is real: `J' = b + (J − b)·exp(β·δd)`, `d' = d + δd` renders
the identical observation (verified to 3e-16). Given that, a *previous* claim
that 78 % of restoration error lies outside the family and is therefore
physically inconsistent was **withdrawn**: a two-pixel counterexample with
`dd = [+0.1, −0.1]` produces an identical render yet leaves 99.69 % of the error
outside the *constant*-`dd` family. Family-external is not observation-inconsistent.

What survives is weaker and conditional: for the fitted water model, a
substantial fraction of a restoration's colour error can be reproduced by a
depth/radiance step that leaves the observation unchanged. That is a statement
about a model class, not a bound on any real scene.

## 8. Provenance and leakage

`artifacts/provenance/` (13 578 files, 6 789 pairs, SHA-256). Directory-level
overlap between splits:

| dataset | train∩val (in/GT/paired) | train∩test | val∩test |
|---|---|---|---|
| UIEB | 0/0/0 | 0/0/0 | 0/0/0 |
| LSUI | 12/14/10 | 10/17/9 | 2/5/2 |
| UFO | 1/1/1 | 3/3/3 | 2/2/2 |

Directory overlap is not proof that a checkpoint consumed those files, and it is
not proof of equal bias in both directions. It is, however, enough to forbid the
word "clean" for these splits. The supplied splits are also not proven to be the
canonical UIEB-800/90 or LSUI-3879/400 (counts differ, no provenance document).

Because the public test splits were inspected during protocol work, every H2/H3
number here is labelled exploratory; none of it is a sealed confirmation.

## 9. Reproduction

* Remote: `/home/user/obsui`, interpreter
  `/usr/local/iCompute/envs/Mamba/bin/python` (Python 3.8.20, torch 2.1.0+cu121).
* Every artifact records the SHA-256 of the script that produced it where the
  script supports it; `artifacts/h3_v2_sources.sha256` pins the H3 sources.
* Commands, per-experiment status and notes: `docs/experiment_ledger.csv`.
* `docs/research_plan.md` records the design and its seven revisions, including
  every defect found after the fact.
* Compute actually spent: well under the 300 A100-hour cap. The only GPU work
  was inference for 21 baseline reruns plus feature extraction; H3, H1 and H2
  are CPU. No training was performed, so the three-seed training study required
  for a final claim was **not** run.

## 10. What would have to be true for this to work

1. The support statistic must carry chroma-error information that survives
   controlling for optical depth and an ensemble's disagreement — H2 v3 says it
   does not, at this sample size, on these two domains.
2. The gate must beat the matched controls (`const_g`, `random_g`, `degrade_g`)
   rather than merely differ from `base` — H3 says it does not.
3. The denominator must be a reproducible SOTA under the same protocol —
   currently unverifiable.

Any one of these failing is enough to stop the claim. All three failed.
The honest summary is that this is a well-instrumented negative result with a
reusable measurement tool, not a method paper.
