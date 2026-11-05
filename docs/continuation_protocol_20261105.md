# Continuation protocol registered 2026-11-05 (post-exploration)

This is not retrospective preregistration. The existing UIEB 89-image test has
already been explored. New runs and their changes are identified below; no
unchanged public split will be called sealed or a fresh blind test.

## Fixed requirements

The original task and its six per-dataset PSNR/SSIM relative-gap limits (each
<=0.05) remain unchanged. A failed or unverified mechanism is not made successful
by copying the base model and renaming it. Official or clean-SOTA status requires
checkpoint and training-split evidence, not agreement with a spreadsheet.

## Repair A: frozen-output diagnostic v2

Observed implementation defects: projection oracle attempted float(length-3
array); chroma_opp used log ratios with EPS=0.05 but apply_chroma omitted that
same offset. Direct round-trip max RGB error was 0.3053414347222976 on a seeded
4x7 input. Correct inverse: G+EPS=(luma+EPS)/denom; subtract EPS after recovering
each channel. This affects every chroma-reconstructed variant, even zero edits.
Legacy four-image results are retained and invalid as mechanism evidence.

Run exactly the same 11 intervention definitions at strength=1 on all 89 stored
UIEB outputs, with historical image-noise fitting. Keep the old artifact. v2
adds per-image records, prediction PNGs, source SHA256, and baseline-protocol
SSIM win5; Gaussian11 is retained and named separately. No quality threshold is
changed. A zero-step self-orbit transform is only a tautological control. Its
oracle version uses references and ignores some feasibility constraints, so it
is not a physically certified ambiguity set or a bound on all possible methods.

Verify the actual imported functions against explicit broadcast objectives and
a zero-intervention chroma round trip before the run. Do not infer the new
scientific outcome from the old buggy output.

## Audit B: actual data and checkpoints

Hash source inputs and references for every available train/val/test split.
Report input-only, reference-only, and paired duplicate counts separately.
Reference duplication alone is not proof of input leakage, and equal split
contamination does not imply equal method bias. Canonical split provenance and
checkpoint training history remain unresolved unless independently established.
No source images or third-party weights are redistributed without licence.

## Diagnostic C: independent controlled detection measurements

Use fresh analytic scenes (seeds 1000..1007), not benchmark references. Use a
linear-domain forward model with separate direct attenuation and backscatter
growth coefficients. Change attenuation, noise, response and observation count;
include matched observed-brightness conditions. Compute local normalized
colour and spatial sensitivity. Verify finite perturbation detectability with
independent Gaussian noise samples and matched-filter AUC; compare nuisance-
projected directions to known-parameter directions, and report model-mismatch
sensitivity separately. These are model-conditional validation of an instrument,
not physical proof for real sRGB photos or validation of enhancer error H1.
All conditions and failures are retained. No data-dependent condition removal.

## Remaining decisive work

H2 must still compare a learned heteroscedastic or sampled uncertainty baseline
with and without support, fit preprocessing on training groups only, and repeat
on a second backbone and held-out degradation families. Existing centred CV
uses held-out target means and cannot be presented as deployed absolute risk.
H3 still requires uncertainty gating, edge/true-detail measures, matched-quality
sweeps, a train/dev/calibration separation and decisive three-seed experiments.
The main performance and reproducibility statuses must remain unverified or
incomplete until corresponding artifacts exist.
