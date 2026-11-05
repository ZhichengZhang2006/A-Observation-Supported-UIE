"""Descriptive paired-image intervals for the repaired H3 diagnostic.

Intervals added after observing results; not confirmatory significance tests.
No change to the fixed scientific or performance acceptance criteria.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
p=Path('artifacts/h3_mechanism_v2.json')
d=json.loads(p.read_text())
n=d['n']
assert n==len(d['names']) and all(len(v)==n for v in d['per_image'].values())
rng=np.random.default_rng(20261105)
index=rng.integers(0,n,(2000,n))
rows=[]
for variant, samples in d['per_image'].items():
    for ref in ['base','const_g']:
        if variant==ref: continue
        for metric in ['psnr','ssim','chroma','luma']:
            delta=np.array([r[metric] for r in samples])-np.array([r[metric] for r in d['per_image'][ref]])
            ci=np.quantile(delta[index].mean(axis=1),[.025,.975])
            rows.append({'variant':variant,'reference':ref,'metric':metric,'mean_paired_delta':float(delta.mean()),'ci95_percentile':[float(x) for x in ci]})
output={'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n_images':n,'bootstrap_replicates':2000,'bootstrap_seed':20261105,'unit':'paired image; exact UIEB cross-split overlap audit found none; near-duplicate scenes unverified','status':'exploratory_post_result_descriptive','not_three_seed_training':True,'comparisons':rows}
Path('artifacts/h3_mechanism_v2_intervals.json').write_text(json.dumps(output,indent=2))
print(json.dumps([r for r in rows if r['variant']=='support_g'],indent=2))
