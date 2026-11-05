# Observation-Supported Underwater Image Enhancement

**The study is complete and the result is negative.** The measurement instrument was built and checked; the claim it was built to support did not survive its own controls. Performance: `unverifiable` (no qualified SOTA denominator). Novelty: `unsupported`. Mechanism: `unsupported`. Reproducibility: `incomplete`. This repository keeps the favourable and unfavourable experiments, the implementation repairs, and the withdrawn claims alongside the numbers that refuted them.

## Research question

Can an observation-conditioned sensitivity/ambiguity measure predict colour or detail risk beyond ordinary uncertainty and degradation, and can a controlled intervention use it without sacrificing restoration quality?

The fixed task remains [the research mandate](<AutoResearch_A_Observation_Supported_UIE.md>). Continuation is documented in [the post-exploration protocol](<docs/continuation_protocol_20261105.md>). Existing public test outputs have already been inspected; they are not sealed confirmatory data.

## Current evidence

| Item | Evidence | What it shows — and what it does not |
| --- | --- | --- |
| 21 local-checkpoint reruns | Seven methods × UIEB/LSUI/UFO, respectively 89/428/120 pairs; [registry](<docs/sota_registry.csv>) | Official checkpoint origin, canonical split identity, or clean published SOTA |
| Source audit | 13,578 source files / 6,789 pairs with SHA256; [inventory](<artifacts/provenance/inventory_summary.json>) and [manifest](<docs/data_manifest.md>) | Unknown checkpoint training history or absence of near-duplicates |
| Controlled detector | 416 condition/direction rows, fresh analytic scenes and independent Gaussian observations; [summary](<artifacts/controlled_detection_v2/summary.json>) | Real-camera physical validity or enhancer-error H1 |
| Instrument repairs | Gain-aware derivatives, rank-aware low-frequency directions and physical interval checks; [v3 diagnostic](<artifacts/instrument_checks_v3_local.json>) | Scientific novelty, calibration, or main performance acceptance |
| H2 v3 (decisive) | 89 UIEB images x 3 backbones and 428 LSUI images x 1 backbone, image-disjoint 5-fold CV, ordinary + 7-model ensemble disagreement vs +support, +shuffled control, +noise control; [artifact](<artifacts/h2_risk_v3.json>) | Support adds no significant R2 over the ordinary baseline (CIs include 0), and is significantly *worse* on LSUI PSNR/SSIM (-0.080/-0.067 R2). Seven noise columns match every effect size. |
| H3 v2 (decisive) | 11 frozen-output variants, 89 UIEB images, matched-strength controls, 2000-draw bootstrap; [artifact](<artifacts/h3_mechanism_v2.json>), [intervals](<artifacts/h3_mechanism_v2_intervals.json>) | The support gate makes chroma **worse** (+1.70 %, CI [+0.0004, +0.0043] on the bad side); constant/random/degradation controls behave the same way; the reference-aware oracle projection loses PSNR, so the anchoring is wrong. |
| H1 v2 | corrected block-level correlations (22 784 blocks, 89 images); [artifact](<artifacts/h1_error_form_v2.json>) | chroma-error rho(support) = +0.072 raw, -0.000 once ordinary statistics are controlled, while forward statistics keep +0.196. |

**Findings:** H2, H3 and the performance gate all fail — see [the research report](<docs/research_report.md>) and [the acceptance report](<docs/acceptance_report.md>) (machine-readable: [JSON](<docs/acceptance_report.json>)). Reproduction landmines and next steps: [handover](<docs/HANDOVER.md>). Historical withdrawn reports are retained in [the archive](<docs/history/20261105_pre_audit/README.md>); their scientific interpretations are superseded, not endorsed.

## Measurement and interpretation

The declared simplified model is `I=g*[b+(J-b)*exp(-beta*d)]+noise`. Local sensitivity profiles a specified nuisance tangent space, not every possible water/camera model. With known fitted gain, the scene derivative is `g*exp(-tau)`, not merely transmission. Real sRGB inputs and fitted residual noise do not automatically satisfy a calibrated sensor likelihood.

The transform `d'=d+dd`, `J'=b+(J-b)*exp(beta*dd)` preserves this model's rendered observation **when the anchor actually fits that observation and constraints remain valid**. This is a conditional symmetry, not a new theorem or a proof that all restored colours are unsupported. A global-depth family is only a subset of compatible scenes; pixel-varying depth gives a direct counterexample to “family-external error implies observation inconsistency”. Projecting an output onto its own family returns itself because zero step is available; no actual input consistency has thereby been checked.

Both radiance floor and ceiling constrain positive steps, and maximum optical depth depends on the deepest pixel. The executable implementation checks include these boundaries. Variance inflation forcing `chi²/dof=1` is residual normalization, **not** empirical calibration or a solution to model mismatch.

## Evaluation protocol and provenance

Primary rerun protocol: RGB, PIL bilinear resize of the entire image to 256×256; no crop/pad; float `[0,1]`; per-image PSNR and skimage SSIM (`win_size=5`, `channel_axis=-1`, `data_range=1`), then arithmetic means. [Data preprocessing](<code/obsui/dataio.py>) and [baseline runner](<code/tools/run_baseline.py>) implement this protocol. The older [metric module](<code/obsui/metrics.py>) has a different resizing/SSIM convention and is not the primary estimator.

MambaCue saved-triptych rescoring is historical evidence only. It is not fresh checkpoint inference; UFO tiles are 240×320 and the original path uses different preprocessing. Those scores are not qualified unified SOTA denominators. Local method weights also lack verified publication/training lineage. CPU FFT fallback invalidates normal GPU latency claims. See [registry caveats](<docs/sota_protocol.md>).

The available UIEB 89 and LSUI 428 test sets are not assumed to be the requested canonical sets. Exact byte duplicates are reported separately for input, reference and paired samples; equal exposure does not guarantee equal method bias. **尚缺受控实拍验证。**

## Reproducing the current diagnostics

Use an existing compatible environment. The audited server interpreter is `/usr/local/iCompute/envs/Mamba/bin/python`; new server outputs remain under `/home/user/obsui`.

```bash
cd /home/user/obsui
python code/tools/check_instruments_v3.py
python code/tools/validate_mechanism_impl.py
python code/tools/controlled_detection_v2.py
python code/tools/make_sota_registry.py
```

These are implementation/controlled diagnostics, not a substitute for the unchanged final requirements: independent split/calibration evidence, ordinary uncertainty comparison, two backbones, held-out degradation families, matched H3 controls, at least three final seeds and all six quality gaps ≤5%. Third-party datasets and weights are not redistributed without verified rights.
