# Handover — observation-supported UIE

Written 2026-11-05. Read this before touching the repository again.

## State in one paragraph

The measurement instrument (support / profiled Fisher information / gauge
interval) is implemented, checked, and reusable. The scientific claim it was
built to support did **not** survive its own controls: H2 does not separate
support from ordinary + ensemble uncertainty, H3 shows the support gate makes
chroma worse, and the performance gate has no qualified SOTA denominator. The
repository is a negative result with good instrumentation, not a method paper.

## What is trustworthy

| artifact | what it establishes | caveat |
|---|---|---|
| `artifacts/baseline_*_*.json` (21) | these 21 checkpoint files, scored under the frozen protocol | publication provenance of each checkpoint unverified |
| `artifacts/mambacue_verify.json` | the frozen protocol is the MambaCue authors' own (≤0.04 dB) | MambaCue tiles only, not a rerun |
| `artifacts/h1_error_form_v2.json` | support carries no independent block-level chroma-error signal | single dataset, block-level, one backbone family |
| `artifacts/controlled_detection_v2/` | the statistic tracks detectability when the model is correct | analytic scenes, no networks |
| `artifacts/h3_mechanism_v2*.json` | the gate hurts chroma; matched controls behave the same way | single dataset, frozen outputs, one strength |
| `artifacts/provenance/` | SHA-256 of every file; cross-split exact duplicates | directory overlap ≠ checkpoint consumption |
| `artifacts/instrument_checks_v3_*.json` | the code does what its equations say | checks code, not science |

## What must not be cited

* `artifacts/h3_mechanism.json` (4-image version) — `apply_chroma` was not the
  inverse of `chroma_opp`; every number in it is computed in a corrupted chroma
  space.
* `artifacts/gauge_error.json` — GT-anchored "best step" over a default water
  model; it measured alignment to the reference, not a bound from the
  observation. Its "78 % of error is physically inconsistent" reading is
  withdrawn (a two-pixel counterexample produces an identical render with 99.69 %
  of the error outside the constant-step family).
* `artifacts/h2.json` — centred-within-image CV with held-out target means;
  scores relative position, not deployable risk, and its support columns carry
  the gain bug.
* Any statement that support is validated as a physical quantity on real data.

## How to reproduce

```bash
ssh school-gpu2
cd /home/user/obsui
PY=/usr/local/iCompute/envs/Mamba/bin/python

# baselines (already done; ~5 min/method on a quiet GPU node)
$PY code/tools/run_baseline.py --method phaseformer --dataset UIEB --split test

# features for H2 (needs predictions/ first)
$PY code/tools/extract_features_v3.py --dataset UIEB --split test --backbone phaseformer --workers 16
$PY code/tools/ensemble_disagreement_v3.py --split test --dataset UIEB

# H2 risk model
$PY code/tools/h2_risk_v3.py --domains UIEB LSUI --seeds 0 1 2 --boot 800 \
    --targets psnr ssim chroma --epochs 250 --out artifacts/h2_risk_v3.json

# H3 mechanism
$PY code/tools/mechanism.py --out artifacts/h3_mechanism_v2.json --save-dir predictions/h3_v2
$PY code/tools/summarize_h3_v2.py
```

Everything runs on CPU except baseline inference. `torch.set_num_threads` is
pinned to 1–2 in the heavy loops: on a shared 96-core host the default pool is
pure contention (Phaseformer was 32× slower).

## Landmines found the hard way

1. **`to_unified` returns uint8 [0,255]; predictions load as [0,1].** Mixing them
   gave PSNR −42 dB and SSIM 0.000 for a whole feature table before anyone
   noticed. `extract_features_v3.py` now asserts `0 < psnr < 60`.
2. **`metrics.py` is legacy.** It uses an aspect-preserving resize plus an 11×11
   Gaussian SSIM. The primary protocol is `skimage` win_size=5 (in
   `run_baseline.score`). Never mix them in one table.
3. **`apply_chroma` / `chroma_opp`** must round-trip to <1e-9; the `EPS = 0.05`
   offset belongs in both directions.
4. **`components(h, w, k)`** returns rank `k−1`; the SVD truncation is required
   or an arbitrary null column enters the basis.
5. **`direction_info`** must include the channel gain (`q = (gain·T/σ)²`); missing
   it silently rescales every global support feature.
6. **ssh + long inline Python** is fragile: multi-line `python -c` inside `ssh`
   loses quoting. Write a file and `scp` it.
7. **`pkill -f <pattern>` inside an ssh command matches the ssh command itself**
   and kills the session (exit 255). List PIDs first, then kill by number.
8. **Two `nohup` copies of a pipeline race on the same output files.** Check
   `ps` before relaunching after a failure.

## Next steps, in the order that would actually settle it

1. **Establish a denominator.** Get publication-verified weights for two methods
   (the repos' release assets, not local `checkpoints/`), rerun on the frozen
   protocol, and only then evaluate the 5 % gate. Without this the performance
   item cannot leave `unverifiable`.
2. **Run H2 on a third domain.** Two domains with 89 and 428 images and R² ≈ 0.01
   cannot distinguish "no information" from "too little data". A power analysis
   before the run, not after.
3. **Train the augmenting backbone.** Every H2 number here uses frozen models
   whose errors were produced without any knowledge of support. The mechanism
   claim is about a model trained *with* the statistic; that experiment was never
   run and is the only one that could rescue H3.
4. **Seal a test split.** The current splits were inspected during protocol work.
   Any final claim needs data that was never looked at.
5. **Decide honestly whether to keep the frame.** If (3) fails, the right output
   is a measurement-tool paper (`support` as a diagnostic for *when a restoration
   cannot be trusted*), not an enhancement method.
