# Shared protocol and host caveats for sota_registry.csv

Every row of `sota_registry.csv` was produced under these conditions; the `notes` column there records only what is specific to that row.

Host caveat: on this shared GPU node, device-side torch.fft raised CUFFT_INTERNAL_ERROR during our runs (reproduced with a length-8 1-D float32 transform in a fresh process). Spectral baselines (Phaseformer, SS-UIE, SFGNet, CDF-UIE, FDCE-Net) were therefore run with their transforms on CPU and the result moved back. The transforms are float32 on either device, so this is an execution-location change, not a numerical-protocol change -- but it does mean the wall-clock times in these artifacts must NOT be compared with published runtimes. torch.set_num_threads(8) was also pinned: the default 48 threads on a shared 96-core host made Phaseformer ~32x slower through oversubscription without changing any metric.

unified_ssim uses skimage.metrics.structural_similarity(win_size=5, channel_axis=-1, data_range=1.0) as the protocol specifies; this reproduces the MambaCue anchor (0.9224 recorded vs 0.9221 measured). The frozen code/obsui/metrics.py implements an 11x11 Gaussian instead and reads ~0.002-0.003 lower; that value is retained per-image in the artifact JSON as ssim_frozen11 for sensitivity checking. PSNR and MSE are identical under either estimator.
