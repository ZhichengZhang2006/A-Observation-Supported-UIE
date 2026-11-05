# Available-data manifest and overlap audit — 2026-11-05

Source: `/home/user/UnderWaterDataset` on `school-gpu2`; experiment workspace `/home/user/obsui`. Counts are **available directory splits**, not a certification of canonical dataset identity. No source images are redistributed here.

## Complete byte-hash inventory

[The inventory script](<../code/tools/inventory_research.py>) recorded 13,578 source files and 6,789 resolved input/reference pairs in 289.477 seconds. [Source-file manifest](<../artifacts/provenance/source_images.csv>) records every input/reference SHA256 and source path. [Paired manifest](<../artifacts/provenance/split_pairs.csv>) SHA256 is `14bc34127319d4e6f26fa6f3c69014a4c9af88dffb71b8b0f362d2825c908e6d`; [summary](<../artifacts/provenance/inventory_summary.json>) timestamp is `2026-11-05T10:10:13Z`.

| Available dataset | Train pairs | Validation pairs | Test pairs |
| --- | ---: | ---: | ---: |
| UIEB | 712 | 89 | 89 |
| LSUI | 3423 | 428 | 428 |
| UFO | 1200 | 300 | 120 |

All nine source folders resolve bijectively in [the pairing audit](<../artifacts/provenance/pairing_and_duplicates.json>): no unmatched inputs or reused reference paths were reported. This pairing fact does not establish canonical split membership. In particular, **89 is not assumed to be UIEB-R90 minus one**, and LSUI 428 is not assumed to be the task's canonical LSUI test.

## Exact cross-split overlaps

Counts are held-out files whose byte-identical SHA256 occurs in the earlier-listed split; the paired column requires the same input **and** reference hashes in one pair. These columns are not additive.

| Dataset | Splits | Input files overlapping | Reference files overlapping | Identical pairs |
| --- | --- | ---: | ---: | ---: |
| UIEB | train / val | 0 | 0 | 0 |
| UIEB | train / test | 0 | 0 | 0 |
| UIEB | val / test | 0 | 0 | 0 |
| LSUI | train / val | 12 | 14 | 10 |
| LSUI | train / test | 10 | 17 | 9 |
| LSUI | val / test | 2 | 5 | 2 |
| UFO | train / val | 1 | 1 | 1 |
| UFO | train / test | 3 | 3 | 3 |
| UFO | val / test | 2 | 2 | 2 |

Every matching path and hash is retained in [the detailed audit](<../artifacts/provenance/pairing_and_duplicates.json>). A zero byte-hash overlap does not rule out re-encoding, near-duplicates, common scenes or checkpoint pretraining exposure. The earlier sampled perceptual-hash screen is not a complete near-duplicate audit and is not promoted to one.

These available directory overlaps are a measurable leakage **risk**. They do not prove that any particular checkpoint trained on these directories, because actual training manifests are unknown. If multiple methods consumed overlapping samples, the resulting bias need not be equal. Future new fits must union input/reference duplicate groups before splitting, or remove overlaps prospectively and label the resulting evaluation subset explicitly. Report full-set and sensitivity results; do not silently delete poorly scoring samples.

## Checkpoints, environment and rights

[Checkpoint inventory](<../artifacts/provenance/checkpoints.json>) records local weight paths, bytes and SHA256. [Environment snapshot](<../artifacts/provenance/environment.json>) records interpreter, packages and GPU details. A hash establishes which bytes were used; it does not establish publication origin, training data or a licence. The baseline source folders have not been verified as original publication checkouts. No third-party source image or weight is redistributed without verified rights.

## Physical limits of the references

Benchmark references are target images, not controlled measurements of water-free scene radiance. Matching one supports reference-based visual quality claims only. Synthetic optical-model checks cannot replace **尚缺受控实拍验证**. The older gauge-error analysis anchors its orbit at a reference with assumed water parameters; it is not an observation-only certificate and its family-external residual is not proof of physical inconsistency.

## Reproduction

```bash
cd /home/user/obsui
/usr/local/iCompute/envs/Mamba/bin/python code/tools/inventory_research.py
```

Canonical file-list/source verification, a complete near-duplicate audit and actual training lineage remain open. The [historical manifest](<history/20261105_pre_audit/data_manifest.md>) is preserved as superseded; its missing-one, equal-bias and observation-only claims are withdrawn.
