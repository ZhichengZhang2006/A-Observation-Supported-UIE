"""Report the summary block of an artifacts/*.json file."""
from __future__ import annotations

import json
import sys

path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    d = json.load(fh)

s = d.get("summary") or d.get("summary_dict") or {}
print(f"file={path} n={d.get('n')}")
for k, v in s.items():
    print(f"  {k:20s} {v:+.4f}" if isinstance(v, float) else f"  {k:20s} {v}")

rows = d.get("rows") or []
if rows:
    for key in ("frac_inside", "dd", "raw_frac_inside", "raw_dd", "width", "censored"):
        vals = [r[key] for r in rows if key in r and isinstance(r[key], (int, float))]
        if not vals:
            continue
        vals_s = sorted(vals)
        n = len(vals_s)
        q = lambda p: vals_s[min(n - 1, int(p * n))]
        print(f"  {key:20s} mean={sum(vals)/n:+.4f} med={q(0.5):+.4f} "
              f"p10={q(0.1):+.4f} p90={q(0.9):+.4f} min={vals_s[0]:+.4f} max={vals_s[-1]:+.4f}")
    for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
        c = sum(1 for r in rows if r.get("frac_inside", 0) > thr)
        print(f"  frac_inside > {thr:.1f}: {c}/{len(rows)}")
