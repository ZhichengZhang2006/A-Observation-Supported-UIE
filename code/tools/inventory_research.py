"""Read-only provenance inventory. All new outputs are under --out-dir.

Hashes every source image in the three available splits and every checkpoint
actually referenced by a completed baseline artifact. GT-only duplication is
not labelled input leakage. Pairing uses the historical resolver but reports
non-bijective or missing matches rather than silently treating it as verified.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obsui.dataio import resolve_pairs, sha256_of, is_image
from PIL import Image


def dump_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='/home/user/UnderWaterDataset')
    ap.add_argument('--project-root', default='/home/user/obsui')
    ap.add_argument('--out-dir', default='artifacts/provenance')
    args = ap.parse_args()
    root, proj, out = Path(args.data_root), Path(args.project_root), Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    records, pairrows, summaries = [], [], []
    for ds in ['UIEB', 'LSUI', 'UFO']:
        for split in ['train', 'val', 'test']:
            d = root / ds / split
            for role in ['input', 'GT']:
                dr = d / role
                files = sorted(p for p in dr.iterdir() if p.is_file() and is_image(p.name)) if dr.is_dir() else []
                for p in files:
                    with Image.open(str(p)) as im:
                        width, height = im.size
                        mode = im.mode
                    records.append(dict(dataset=ds, split=split, role=role,
                                        name=p.name, path=str(p), bytes=p.stat().st_size,
                                        sha256=sha256_of(str(p)), width=width, height=height, mode=mode))
                print('%s/%s/%s n=%s' % (ds, split, role, len(files)), flush=True)
            if (d/'input').is_dir() and (d/'GT').is_dir():
                pairs = resolve_pairs(str(d/'input'), str(d/'GT'), ds+'-'+split)
                ref_counts = Counter(p.ref_path for p in pairs.pairs)
                summaries.append(dict(dataset=ds, split=split, n_pairs=len(pairs),
                                      notes=pairs.notes, nonbijective_refs={k:v for k,v in ref_counts.items() if v>1}))
                for p in pairs.pairs:
                    pairrows.append(dict(dataset=ds, split=split, name=p.name,
                                         input_path=p.input_path, ref_path=p.ref_path))
    fields = ['dataset','split','role','name','path','bytes','sha256','width','height','mode']
    with open(out/'source_images.csv', 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
    bypath = {r['path']:r for r in records}
    for p in pairrows:
        p['input_sha256']=bypath[p['input_path']]['sha256']
        p['ref_sha256']=bypath[p['ref_path']]['sha256']
    with open(out/'split_pairs.csv', 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(pairrows[0])); w.writeheader(); w.writerows(pairrows)
    duplicates = []
    for ds in ['UIEB','LSUI','UFO']:
        for a,b in [('train','val'),('train','test'),('val','test')]:
            rec = {'dataset':ds, 'splits':[a,b]}
            for role in ['input','GT']:
                lhs = defaultdict(list)
                for r in records:
                    if r['dataset']==ds and r['split']==a and r['role']==role:
                        lhs[r['sha256']].append(r['path'])
                hits=[]
                for r in records:
                    if r['dataset']==ds and r['split']==b and r['role']==role and r['sha256'] in lhs:
                        hits.append({'heldout':r['path'],'train_or_other':lhs[r['sha256']], 'sha256':r['sha256']})
                rec[role]={'n_heldout_files':len(hits),'matches':hits}
            lhs={(p['input_sha256'],p['ref_sha256']) for p in pairrows if p['dataset']==ds and p['split']==a}
            both=[p['name'] for p in pairrows if p['dataset']==ds and p['split']==b and (p['input_sha256'],p['ref_sha256']) in lhs]
            rec['identical_paired_samples'] = both
            duplicates.append(rec)
    dump_json(out/'pairing_and_duplicates.json', dict(pairing=summaries,duplicates=duplicates,
              caveat='Exact byte hashes only. Equal references alone do not prove identical input or checkpoint training contamination. Sequence/perceptual duplicates need separate audit.'))
    checkpoints=[]
    for art in sorted((proj/'artifacts').glob('baseline_*.json')):
        d=json.loads(art.read_text())
        ck=d.get('checkpoint')
        if ck:
            ck=Path(ck)
            checkpoints.append(dict(artifact=art.name,method=d.get('method'),dataset=d.get('dataset'),
                                   checkpoint=str(ck),exists=ck.is_file(),
                                   sha256=sha256_of(str(ck)) if ck.is_file() else None,
                                   bytes=ck.stat().st_size if ck.is_file() else None,
                                   origin_status='local_checkpoint_publication_and_training_provenance_unverified',
                                   elapsed_sec=d.get('elapsed_sec'),n=d.get('n_scored')))
    dump_json(out/'checkpoints.json', checkpoints)
    env=dict(python=sys.version, executable=sys.executable, platform=platform.platform())
    for cmd,key in [([sys.executable,'-m','pip','freeze'],'pip_freeze'),(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],'gpu')]:
        r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        env[key]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
    dump_json(out/'environment.json',env)
    dump_json(out/'inventory_summary.json',dict(n_source_files=len(records),n_pairs=len(pairrows),
              manifest_sha256=sha256_of(str(out/'split_pairs.csv')),elapsed_sec=time.time()-start,
              recorded_wall_time=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
              original_data_license='not yet independently verified; no source images redistributed'))
    print('inventory complete: %s files, %s pairs' % (len(records),len(pairrows)),flush=True)

if __name__=='__main__':
    main()
