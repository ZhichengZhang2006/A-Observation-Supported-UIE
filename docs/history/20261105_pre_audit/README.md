# Observation-Supported Restoration

**Recoverability-aware underwater image enhancement — what a single image can and
cannot tell you about its own colours.**

This repository asks one question and answers it with measurements rather than
architecture:

> Given an underwater image and *no* ground truth, how strongly does the
> observation constrain the colour and structure of the scene behind it — and
> does knowing that change how we should restore?

The short answer the experiments support: **structure is constrained, colour is
not, and the constraint that exists is worth less than the colour prior every
method already uses.** The repository's job is to make that statement measurable
per image, per region and per method, instead of asserting it.

---

## The two measurement objects

Forward model, per channel $c$:

$$ I_c = g_c\left(J_c T_c + b_c (1 - T_c)\right) + n_c, \qquad T_c = e^{-\tau_c}, \quad \tau_c = \beta_c d $$

with water parameters $\psi = (\beta, g, b)$, scene $J$, depth field $d$, and
sensor noise $n \sim \mathcal N(0, \Sigma_n(I))$.

**Object 1 — local observation sensitivity.** For a unit perturbation direction
$v$ in scene space,

$$ q_v = \left\lVert \Sigma_n^{-1/2} D_J G_\psi\, v \right\rVert_2^2 $$

With nuisance parameters profiled out, the reported quantity is the
*leverage-corrected* $q_v^{\text{eff}} = q_v (1 - h)$ where $h$ is the hat-matrix
leverage of direction $v$ on the water-parameter Jacobian. A direction that the
water fit can absorb is a direction the data does not constrain, and $h$ measures
exactly that absorption.

**Object 2 — the gauge family.** The set of scenes that reproduce the *same*
observation bit-for-bit. Its width, not its existence, is the measurement:

$$ \mathcal F(J, \psi) = \left\{ J'(dd) : J'_c = b_c + (J_c - b_c)e^{\beta_c\, dd},\; dd \in \mathcal A(J,\psi) \right\} $$

where $\mathcal A$ is the physically admissible interval set by dynamic range,
non-negativity, non-negative depth and a maximum optical depth.

---

## The result that organises everything else

**$\mathcal F$ is an exact symmetry, not an approximation.** Raising the optical
depth by $dd$ while lifting every channel's signal above the veiling light by
$e^{\beta_c dd}$ leaves the rendered image *analytically unchanged*:

$$ b_c + (J'_c - b_c)e^{-\beta_c(d + dd)} = b_c + (J_c - b_c)e^{\beta_c dd}e^{-\beta_c d}e^{-\beta_c dd} = b_c + (J_c - b_c)e^{-\beta_c d} $$

Measured residual: $3\!\times\!10^{-16}$, i.e. floating-point exact
(`code/tools/test_gauge.py`, G1).

Because $\beta_c$ differs per channel, the required lift differs per channel —
**the ambiguity is chromatic by construction**, and it is the precise sense in
which a restored colour is "unsupported".

Three consequences, each measured rather than assumed:

1. **Projection onto the family is the identity.** $dd=0$ is admissible and
   reproduces the input exactly, so the nearest consistent scene to any
   restoration is itself. Consistency-based losses, gates and projections therefore
   cannot correct colour (`code/tools/mechanism.py`, `project` row).
2. **The positive side is closed, the negative side is open.** Real photographs hit
   the dynamic-range limit in the brightening direction, so $dd>0$ is essentially
   always inadmissible; the darkening direction is limited only by non-negative
   depth and is available for nearly every image. A washed-out restoration and a
   saturated one are equally consistent with the observation.
3. **Darker scenes admit more colour ambiguity, not less.** The binding constraint
   is $J'_c \le 1$, so headroom — not brightness or contrast — sets the budget.
   This is a falsifiable prediction that runs opposite to what any
   brightness/contrast statistic would predict, and it holds
   (`test_gauge.py`, G3: `dd_pos_max` 1.902 → 0.702 → 0.102 → 0 → 0 as scene level
   rises).

**What this does not claim.** It does not claim a single image cannot be enhanced.
It claims that any *specific* colour correction supplies information from outside
the observation — a scene prior, a learned colour distribution, or noise. Those can
all be useful. None of them is a measurement of the scene, and a method should not
report it as one.

---

## What is in here

| Path | What it does |
|---|---|
| `code/obsui/support.py` | Object 1: concentrated-likelihood water fit, Jacobian, leverage, $q_v$, $q_v^{\text{eff}}$, profile-likelihood widths |
| `code/obsui/gauge.py` | Object 2: the exact family, admissible range, closed-form best step, family PSNR spread |
| `code/obsui/controlled.py` | Seeded forward-degradation track, one physical axis at a time, with a real detectability test |
| `code/obsui/metrics.py` | Frozen metric implementations (PSNR / SSIM / MSE, per-image then averaged) |
| `code/obsui/dataio.py` | Frozen preprocessing: whole image → 256×256, bilinear, no crop, no pad |
| `code/tools/` | Experiments, baselines, diagnostics, artifact reporting |
| `docs/` | Literature matrix, novelty audit, measurement tables, registry |
| `artifacts/` | Per-image JSON evidence for every number the report cites |

---

## The evaluation protocol is frozen, and verified against a reference

Everything is scored by resizing the **whole image** to 256×256 with bilinear
interpolation — no cropping, no padding, one resampling for every method — then
PSNR/SSIM with `skimage` (`win_size=5`, `channel_axis=-1`, `data_range=1.0`),
per image, then averaged.

The protocol is not asserted; it is *reproduced*. Scoring MambaCue's own saved
prediction tiles gives

| dataset | measured | the run's recorded value | Δ |
|---|---|---|---|
| UIEB (89) | 24.4762 / 0.9221 | 24.4629 / 0.9224 | 0.013 dB |
| LSUI (428) | 31.1957 / 0.9351 | 31.2323 / 0.9357 | 0.037 dB |
| UFO-120 (120) | 28.5965 / 0.8780 | 28.6112 / 0.8789 | 0.015 dB |

Agreement within 0.04 dB on all three datasets is why the frozen protocol is taken
to be the reference method's *own* protocol rather than an approximation of it. The
5% acceptance gate is computed against these frozen numbers, so getting the
protocol wrong would have made every gap meaningless.

**A refused shortcut.** Third-party prediction folders circulating for these
datasets score 2.3 dB *below* the reported value on UIEB while scoring 0.6 dB
*above* it on UFO-120 — wrong in both directions, which no single rescaling
explains. Rather than quote them, every baseline in `docs/sota_registry.csv` with a
number in the `unified_*` columns was re-run from official weights here. Where an
official weight does not exist on this machine, the row says `not_run` and gives
the reason; it does not say zero.

---

## Cost and honesty notes

The metric code pins BLAS threads to 1 at import: on a 96-core node the normal
equations for a 9-parameter fit were being spread over 96 threads and ran **18×
slower** than single-threaded. The fix is in `code/obsui/__init__.py` and matters
because it changes whether a per-image analysis is usable at all.

Water-parameter fits use `noise_mode="misfit"`, which inflates the noise floor by
the observed misfit. With sensor noise alone the model explains a few percent of
the residual variance, $\chi^2/\text{dof}$ reaches $10^5$–$10^7$, every direction
appears infinitely determined and the support map collapses to a constant — an
instrument artefact that produced four identical ablation numbers before it was
caught. Inflating to $\chi^2/\text{dof} = 1$ makes $q_v$ mean "information per unit
of *total unexplained* variance", which is the only version that is honest on real
data. Which mode produced a number is recorded with the number.

Reference images in UIEB/LSUI are human-selected or algorithmically produced, not
water-free truth. Every conclusion that touches a reference is filtered through
that caveat, and the leakage screen (exact + perceptual hash) is in
`docs/data_manifest.md`.

---

## Reproducing

```bash
# metrics self-test -- no data needed
python -c "import sys; sys.path.insert(0,'code'); from obsui import metrics; metrics._selftest()"

# the gauge identity (14 assertions, no GPU, no data)
python code/tools/test_gauge.py

# the support estimator (14 assertions, synthetic)
python code/tools/test_support.py
```

Both test suites are CPU-only and complete in under a minute. They exist because
the conclusions rest on instrument correctness: three of the early "findings" in
this project turned out to be implementation errors (a 3-parameter model posing as
a 1-parameter family, a wrong denominator, and a profiler objective that let pixels
quietly leave the fit), and each was caught by a test that asserted the *physics*
rather than the code's own output.
