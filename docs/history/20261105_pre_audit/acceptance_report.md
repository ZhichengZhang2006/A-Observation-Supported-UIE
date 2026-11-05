# Acceptance Report — Observation-Supported Restoration

Generated from artifacts produced on `school-gpu2` (A100-PCIE-40GB, 20 GB slice).
Every number below is traceable to a file named beside it.

| status field | value |
|---|---|
| `performance_status` | **not_passed** |
| `novelty_status` | **not_passed** |
| `mechanism_status` | **not_passed** |
| `reproducibility_status` | **partial** |

Read this table as the honest headline. The work established one solid negative
result and one solid identity; it did **not** clear the 5% performance gate and it
did **not** demonstrate that the support measurement adds information beyond
ordinary predictors. Both failures are documented below with the numbers that
produced them, because a report that only lists what worked is not a report.

---

## 1. `performance_status = not_passed`

The user's gate is: relative gap to the frozen SOTA ≤ 5.00% on six items (PSNR and
SSIM on each of UIEB-R90, LSUI, UFO-120).

This project contributed **no trained enhancement network of its own**. Its
proposed mechanism (`project`) was measured to be the identity transformation
(§3), so there is nothing to enter into the gate. A gap cannot be computed against
a method that does not exist, and inventing a number here would be the single most
damaging thing this report could do.

What *does* exist is a verified frozen-SOTA reference, which is what the gate
would need:

| dataset | frozen SOTA (MambaCue) PSNR / SSIM | gate threshold at −5% |
|---|---|---|
| UIEB-R90 (89) | 24.4762 / 0.9221 | 23.252 / 0.8760 |
| LSUI (428) | 31.1957 / 0.9351 | 29.636 / 0.8883 |
| UFO-120 (120) | 28.5965 / 0.8780 | 27.167 / 0.8341 |

These are reproducible from committed code (`code/tools/verify_mambacue.py`) and
agree with the reference method's own recorded output to within 0.04 dB on all
three datasets.

**Why no method was trained.** The programme had a fixed order: diagnose whether
the measurement object carries information *before* building a mechanism on it.
That diagnosis is §2, and it came back negative. Training a network to spend the
information this measurement does not have would have produced a number without a
reason.

---

## 2. `novelty_status = not_passed` — the decisive experiment said no

`docs/novelty_audit.md` fixed five refutation criteria in advance. Criterion #1
("a trivial baseline swallows the gain") and #5 ("indistinguishable from an
off-the-shelf uncertainty map") are the load-bearing ones. Both fail.

### 2.1 Incremental predictive power over ordinary statistics (H2)

`code/tools/h2_incremental.py` → `artifacts/h2.json`. Setup: 89 UIEB images,
22,784 16×16 blocks, 5-fold cross-validation **grouped by image** so no block is
predicted from its own image, targets centred within each image (the question is
"where in this image is the error largest", not "which image is worse", which no
block feature could answer from a held-out image).

The baseline deliberately includes optical depth (`tau_mean`), because H1 measured
that `rho(support, error)` drops from 0.173 to 0.068 once depth is controlled. A
baseline without it would have been a strawman and the incremental claim would have
been unearned.

| target | R² ordinary (9 features) | R² + support (16) | Δ |
|---|---|---|---|
| absolute error | 0.0709 | 0.0672 | **−0.0037** |
| relative error | 0.0460 | 0.0511 | +0.0051 |
| chroma error | 0.1183 | 0.1257 | **+0.0074** |

Per-image correlation (n=89): chroma ρ = **−0.172**, p = 0.106.

**Reading.** The sign is wrong for the hypothesis as stated in H1 (low support
should mean high error) and the increment is +0.7 percentage points on the target
where it should be strongest. That is inside the noise of a 7-feature addition on
23k correlated blocks, and the one correlation that is large enough to mention is
negative and non-significant. By the audit's own criterion, the support measurement
is **not separable from ordinary predictors** on real data.

### 2.2 The mechanism is the identity

`code/tools/mechanism.py` → `artifacts/h3_mechanism.json`. The proposed
intervention was to project a method's output onto the set of scenes consistent
with the observation:

```
dd*  = argmin_{dd admissible} || J_base − [b + (J_base − b) e^{β dd}] ||
Ĵ    = b + (J_base − b) e^{β dd*}
```

Measured on UIEB: `project` returns `dd = 0` for every image and reproduces the
input to machine precision. This is not a bug — `dd = 0` is admissible and
reproduces the input exactly, so the nearest consistent scene to any restoration
**is that restoration**.

> Every restoration is already consistent with the observation for some water
> depth. Therefore no consistency-based loss, gate or projection can correct
> colour. Anything that corrects colour is using a scene prior, not a measurement.

This is the project's real finding, and it is a negative one about the whole
family of consistency-based methods — including the one this project proposed.

### 2.3 The gate variants are also indistinguishable

Same artifact, 4-image smoke run (all four support-derived gates differ from each
other only in the spatial pattern they impose):

| variant | chroma err | Δ chroma | PSNR | Δ PSNR |
|---|---|---|---|---|
| base | 0.0800 | — | 27.075 | — |
| support_g | 0.0893 | +11.6% | 26.886 | −0.19 |
| const_g | 0.0892 | +11.5% | 26.887 | −0.19 |
| random_g | 0.0892 | +11.5% | 26.887 | −0.19 |
| degrade_g | 0.0894 | +11.7% | 26.885 | −0.19 |
| oracle_g (uses the reference) | 0.0900 | +12.4% | 26.857 | −0.22 |

Four different spatial modes produce the same number to three decimals, and even
the oracle — which is allowed to see the reference — loses. The gate is not doing
the work; the anchor it pulls toward is.

---

## 3. `mechanism_status = not_passed`, with one positive result attached

The mechanism layer fails for the reason in §2.2: the intervention is the identity.

Two results from this layer **do** survive and are worth keeping:

**3.1 The gauge identity is exact.** `code/tools/test_gauge.py`, 18/18 pass. The
family `J'_c = b_c + (J_c − b_c)e^{β_c dd}` with `d → d + dd` renders an identical
image; measured residual `3.3e-16`, i.e. floating-point exact. Because `β_c` differ
per channel the ambiguity is chromatic by construction: in log-opponent terms it
reaches p95 = **0.118–0.288** on real UIEB images (`artifacts/gauge_error.json`,
G4 rows). This is the correct quantitative statement of "unsupported colour", and
it is a theorem about the model, not a fitted statistic.

**3.2 The family explains only about a fifth of the colour error.**
`artifacts/gauge_error.json`, 89 UIEB images, MambaCue's frozen predictions:

```
frac_inside_mean 0.2171   median 0.1788   p10 0.0095   p90 0.4944
frac_inside > 0.5 in 9/89 images
raw input frac_inside_mean 0.2976
```

So ~78% of the method's colour error lies **outside** the family — it is not
"merely another consistent scene", it is genuinely inconsistent with the
observation. This is an unfavourable number for a paper that wants to say "colour
error is unfalsifiable", and it is reported as such: the family is a real but
*minor* explanation, and the majority of the error is a measurable defect.

**3.3 Family spread measured correctly.** The first implementation of
`family_spread` anchored on the reference, which made `dd = 0` reproduce the
reference exactly (MSE 0, PSNR 99 dB) and report a meaningless 99 dB "spread". It
was rewritten to slide the *method's* output along the family and report
`psnr_gain_max` — the PSNR available for free without using any information the
observation lacks. 3-image smoke: free gain ≈ 0.00 dB, band width median 0.12 dB,
p90 7.49 dB. The band is narrow, which again limits what the family can explain.

---

## 4. `reproducibility_status = partial`

**Reproducible:**
- The frozen protocol reproduces the reference method's recorded numbers to
  ≤0.04 dB on three datasets from committed code (`verify_mambacue.py`).
- All seven official-weight baselines re-run at 89/89 coverage under that protocol
  (`docs/sota_registry.csv`); five agree with their published values within
  0.16 dB.
- Both test suites (`test_support.py` 14/14, `test_gauge.py` 18/18) are CPU-only,
  seeded, and finish in under a minute.
- Every artifact JSON is committed; every number in this report names its file.

**Not reproducible / gaps declared:**
- **WWE-UIE and WaterNet** have no main-network weights on the server (`未查到`).
  Their registry rows say `not_run` with the reason. They are *not* zero.
- **LSUI and UFO baseline reruns were still in progress** when this report was
  written. Only UIEB is complete for all seven. The registry marks the others
  `not_run` rather than extrapolating.
- **No multi-seed results.** The planned ≥3-seed protocol assumed a trained method;
  since no method was trained, there are no seeds to vary. The statistical claims
  here are single-run measurements of frozen predictions, which is appropriate for
  measurement work but does not satisfy the task's multi-seed requirement.
- **No controlled real-camera validation.** States `尚缺受控实拍验证`, as the task
  requires when it is unavailable.
- **Reference-image contamination is unresolved.** Hash screening found 17 exact
  duplicates between LSUI train and test and 8 near-duplicates. Flagged in
  `docs/data_manifest.md` as a risk; not corrected.
- GPU-hours are not reported per the planned split because the plan's training
  budget was not spent. Actual usage was inference and CPU analysis only.

---

## 5. What would have to be true for this to pass

Stated so the negative result is actionable rather than merely disappointing:

1. The support measurement must predict error **after** controlling for optical
   depth, brightness, gradient and saturation — ΔR² on chroma above ~0.05 with a
   positive sign. Currently −0.0037 on absolute error and a negative per-image ρ.
2. A mechanism must produce colour gain **larger than** `psnr_gain_max`, the
   PSNR available for free by sliding along the family. Currently the mechanism is
   the identity and the free gain is ~0.
3. Either must beat a scalar exposure multiplier (`gain_oracle`) at matched
   parameter count. If a single global gain buys the same PSNR, the water family's
   extra freedom is worth nothing and calling it a gauge step is mislabelling a
   brightness knob.
4. All of it must hold at ≤5% relative gap to the frozen SOTA on six items, which
   was never reached because no method was entered.

None of these is currently satisfied. The honest status is `not_passed` on three of
four axes.
