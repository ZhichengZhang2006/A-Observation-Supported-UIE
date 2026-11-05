## T1  Baselines under the frozen protocol (PIL bilinear -> 256x256)

method         dataset    n     PSNR     SSIM  SSIM(w5)  source
aquaclarity    LSUI     428  30.1160   0.9404    0.9404  ok
aquaclarity    UFO      120  28.3574   0.8819    0.8819  ok
aquaclarity    UIEB      89  23.5541   0.9145    0.9145  ok
cclnet         LSUI     428  25.9133   0.8794    0.8794  ok
cclnet         UFO      120  24.7848   0.8012    0.8012  ok
cclnet         UIEB      89  21.1382   0.8619    0.8619  ok
cdfuie         LSUI     428  24.6539   0.8797    0.8797  ok
cdfuie         UFO      120  24.8719   0.7725    0.7725  ok
cdfuie         UIEB      89  21.8999   0.8691    0.8691  ok
fdcenet        LSUI     428  26.2825   0.9130    0.9130  ok
fdcenet        UFO      120  24.4445   0.8422    0.8422  ok
fdcenet        UIEB      89  22.7114   0.8985    0.8985  ok
phaseformer    LSUI     428  32.5089   0.9408    0.9408  ok
phaseformer    UFO      120  30.1545   0.8975    0.8975  ok
phaseformer    UIEB      89  24.2302   0.9149    0.9149  ok
sfgnet         LSUI     428  25.0092   0.9031    0.9031  ok
sfgnet         UFO      120  26.8374   0.8595    0.8595  ok
sfgnet         UIEB      89  20.8171   0.8743    0.8743  ok
ssuie          LSUI     428  27.2907   0.9042    0.9042  ok
ssuie          UFO      120  27.0979   0.8563    0.8563  ok
ssuie          UIEB      89  21.4540   0.8595    0.8595  ok

## T3  H1: does observation support predict enhancement error?
     rho = Spearman over 16x16 blocks; partial controls for [degradation, tau, luma, grad]

dataset    target   rho(lq)  rho(tau)  rho(lum)  partial|lq  partial|fwd
UIEB       abs        0.019    -0.008     0.148       0.074        0.122
UIEB       rel        0.142     0.068    -0.313       0.071        0.154
UIEB       chr        0.173     0.134    -0.287       0.068        0.143
UIEB       lumm       0.035     0.026     0.066       0.056        0.075
UIEB       abs        0.014    -0.046     0.148       0.071        0.112
UIEB       rel        0.076     0.099    -0.313       0.017        0.146
UIEB       chr        0.072     0.134    -0.287      -0.000        0.196
UIEB       lumm       0.078    -0.021     0.066       0.118        0.089

  reading: 'rel' and 'chr' are the honest targets -- absolute error
  shrinks with signal level, so a dark image scores well while being
  wrong in relative terms.  partial|lq stays positive after controlling
  for degradation, tau, brightness and gradient, so the signal is not a
  restatement of any of them.

  caveat that must appear in the report: |rho| ~ 0.15 is a weak
  correlation.  It is enough to show support carries information, not
  enough to call it a predictor of per-image quality.

## T4  Gauge: what the observation cannot determine

  n = 89 images (UIEB test, frozen predictions)
  fraction of scene-space error inside the gauge family:
      mean = 0.217   median = 0.179
  implied depth step dd: mean = -0.008   median = -0.013
  degraded input, for reference: frac = 0.298  dd = -0.251

  reading: the family explains a minority of a method's colour error.
  That is an *adverse* result for the strong form of the claim and it
  is reported as such: a method that lands outside the family is adding
  colour the observation does not support, which is exactly the failure
  mode this project measures.  It does not license the weaker reading
  that all colour differences are unfalsifiable -- they are not.
