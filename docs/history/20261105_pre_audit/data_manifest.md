# Data manifest and leakage audit

Every number in this project is downstream of these files. This document records
where they came from, what they hash to, and what is wrong with them — including
the part that weakens our own evaluation.

All measurements below were taken on `school-gpu2` against
`/home/user/UnderWaterDataset` and are reproducible from the commands shown.

## 1. Splits used

| dataset | role | train | val | test | GT present |
|---|---|---|---|---|---|
| UIEB | primary | 712 | 89 | 89 | yes (server copy only) |
| LSUI | primary | 3423 | 428 | 428 | yes |
| UFO-120 | primary | 1200 | 300 | 120 | yes |
| EUVP-d | not used | 4440 | 555 | 555 | yes |
| CH60 | no-reference only | — | — | (test dir empty) | n/a |
| U45 | no-reference only | — | — | 45 | n/a |

**UIEB-R90 is 89 images here, not 90.** The task specification says 800/90. The
server test split contains 89 reference pairs; the tenth-decade holdout is one
short. We report `n=89` everywhere rather than padding or re-splitting, because
silently using a different set than the one the frozen SOTA was measured on would
make the ≤5% gap meaningless.

**The local copy of UIEB is unusable for evaluation.** `fsdownload/SFGNet-main/data/UIEB`
has 712 train inputs but only 170 train GTs, and its `test/` and `val/` directories
are empty. UFO's `test/` is empty locally too. All reference images therefore come
from the server copy, pulled with `tar cf - GT | tar xf -` (verified: 89/89, 428/428,
120/120 stems match the frozen prediction files after normalising the `_img_`
suffix and the `.jpg`/`.png` extension difference).

## 2. Leakage between splits — measured, not assumed

Method: SHA-1 over file bytes for exact duplicates. The task also asks for
perceptual hashing; `code/tools/audit_assets.py` implements a dhash with
hamming ≤ 4 and sampled 400×200 pairs, but the exact-hash numbers below are the
ones that matter, because a byte-identical file cannot be a coincidence.

### LSUI: 17 of 428 test references are byte-identical to a training reference

```
train_unique=3399  train_total=3423   (24 duplicate hashes within train)
test_unique=425    test_total=428     (3 duplicate hashes within test)
shared_hashes=14   →  17 test files affected
```

Distribution of shared hashes over test files: 13 hashes appear once, one hash
(`f312c89d…`) appears **three** times — `test/GT/1052.jpg`, `test/GT/1063.jpg` and
`test/GT/1085.jpg` are the same file as `train/GT/1098.jpg`. Verified directly:

```
$ sha1sum train/GT/1098.jpg test/GT/1052.jpg test/GT/1063.jpg test/GT/1085.jpg
f312c89d55fa0dc6d44e686cd7e93899af54fd6d  train/GT/1098.jpg
f312c89d55fa0dc6d44e686cd7e93899af54fd6d  test/GT/1052.jpg
f312c89d55fa0dc6d44e686cd7e93899af54fd6d  test/GT/1063.jpg
f312c89d55fa0dc6d44e686cd7e93899af54fd6d  test/GT/1085.jpg
```

### UFO-120: 1 near-duplicate, 0 exact duplicates crossing train/val

`set_f384.jpg` (val) matches `set_u324.jpg` (train) exactly under dhash, not under
SHA-1. Recorded as a near-duplicate only.

### UIEB: unknown

The audited split had an empty `val/`, so `train_vs_val` could not be computed
(`n_b=0`). The server copy has 89 val images; this comparison has **not** been
re-run against it. Recorded as a gap, not as "clean".

### What this does to our conclusions

It does **not** affect the gauge theorem, the 18/18 gauge tests, or the H2/H3
results, because none of those depend on LSUI test performance. It **does** bound
any future claim about LSUI generalisation: 17 images (4.0% of the test split)
were available during training, so LSUI numbers are optimistic by an amount we
have not measured. Any method comparison on LSUI inherits the same optimism
equally, which is why the baseline sweep is still internally valid — but the
absolute LSUI numbers are not clean held-out numbers and must not be quoted as
such.

## 3. Reference images are not ground truth

UIEB/LSUI/UFO "GT" images are either human-selected or algorithmically enhanced
outputs, not water-free radiance measured in the field. Consequences we accept
and state:

- A method can score well by matching the *aesthetic* of the reference set rather
  than by recovering the scene.
- `gauge_error.py` fits the gauge family from the **observation and water model
  only**, never from the reference, precisely so that "fraction of error inside
  the family" is not circular. The reference is used afterwards, to measure, not
  to define.
- The project does not claim these references establish physical correctness.
  Where the task asked for real calibrated data and none exists, the report says
  **尚缺受控实拍验证** rather than substituting synthetic evidence.

## 4. Reproducing this audit

```
# exact-duplicate hashes per split
sha1sum train/GT/*.jpg | awk '{print $1}' | sort > /tmp/tr.txt
sha1sum test/GT/*.jpg  | awk '{print $1}' | sort > /tmp/te.txt
comm -12 <(sort -u /tmp/tr.txt) <(sort -u /tmp/te.txt) | wc -l

# full tree / split-resolution / dhash report
python code/tools/audit_assets.py --root <data-root> --out artifacts/audit.json
```

Manifest of the exact evaluation files (89 + 428 + 120 stems, with the protocol
they were resized under) is in `docs/measurement_tables.md` T1 and in each
`artifacts/baseline_*.json`, which stores per-image PSNR/SSIM so any aggregate can
be recomputed from the rows rather than trusted.
