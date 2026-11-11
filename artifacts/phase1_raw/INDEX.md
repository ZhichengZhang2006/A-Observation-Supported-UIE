# Phase-1 raw run records (E0-E3)

Every number below is copied unmodified from the run that produced it.
`result.json` holds the full per-image val/test rows, the whole validation
curve, and the branch/gate diagnostics; `train.log` is the run's own stdout.
Derived statistics live in the report and can be recomputed from these files.

| tag | arm | dataset | updates | val PSNR | val SSIM | test PSNR | test SSIM | GPU h | new params |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| e0_LSUI_10000 | E0 | LSUI | 10000 | 31.0033 | 0.936106 | 31.2006 | 0.935291 | 6.34 | 0 |
| e0_UIEB_10000 | E0 | UIEB | 10000 | 23.3506 | 0.908545 | 23.8572 | 0.916103 | 3.92 | 0 |
| e1_LSUI_10000 | E1 | LSUI | 10000 | 31.0105 | 0.936132 | 31.2009 | 0.935286 | 5.74 | 194432 |
| e1_UIEB_10000 | E1 | UIEB | 10000 | 23.0706 | 0.906499 | 23.6563 | 0.916129 | 4.00 | 194432 |
| e2_LSUI_10000 | E2 | LSUI | 10000 | 31.0089 | 0.936093 | 31.2011 | 0.935300 | 4.24 | 184192 |
| e2_UIEB_10000 | E2 | UIEB | 10000 | 23.1390 | 0.907234 | 23.5938 | 0.914639 | 4.07 | 184192 |
| e3_UIEB_10000 | E3 | UIEB | 10000 | 23.0100 | 0.905684 | 23.4359 | 0.913443 | 4.05 | 251264 |

## Untouched start point (published weights)

| tag | start val PSNR | start val SSIM | start test PSNR | start test SSIM |
|---|---:|---:|---:|---:|
| e0_LSUI_10000 | 31.0116 | 0.936212 | 31.2167 | 0.935176 |
| e0_UIEB_10000 | 23.8255 | 0.912363 | 24.4604 | 0.921924 |
| e1_LSUI_10000 | 31.0116 | 0.936212 | 31.2167 | 0.935176 |
| e1_UIEB_10000 | 23.8255 | 0.912363 | 24.4604 | 0.921924 |
| e2_LSUI_10000 | 31.0116 | 0.936212 | 31.2167 | 0.935176 |
| e2_UIEB_10000 | 23.8255 | 0.912363 | 24.4604 | 0.921924 |
| e3_UIEB_10000 | 23.8255 | 0.912363 | 24.4604 | 0.921924 |
