"""Implementation checks, not scientific or performance acceptance.

Checks actual imported functions against a direct objective and round trips.
No thresholds in the user research contract are altered by these checks.
"""
import json
import os
import sys
import importlib.util
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
spec = importlib.util.spec_from_file_location("mechanism", os.path.join(os.path.dirname(__file__), "mechanism.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
rng = np.random.default_rng(1701)
rows = []
for h, w in [(4, 7), (17, 19), (31, 23)]:
    J = rng.random((h, w, 3))
    beta = np.array([1.155, 1.006, .839])
    b = np.array([.1, .18, .25])
    grid = np.linspace(-1.2, 1.2, 401)
    orbit = b + (J[:, :, None, :] - b) * np.exp(grid[None, None, :, None] * beta)
    for target in [None, rng.random(J.shape)]:
        T = J if target is None else target
        cost = ((orbit - T[:, :, None, :]) ** 2).sum(axis=(0, 1, 3))
        r = m.project_onto_family(J, beta, b, target=target)
        expected = int(cost.argmin())
        err = abs(r['cost_at_dd'] - float(cost[expected]))
        assert abs(r['dd'] - grid[expected]) < 1e-12, (r['dd'], grid[expected])
        assert err < 1e-8, err
        rows.append({'shape': [h, w], 'oracle': target is not None, 'cost_error': err})
    back = m.apply_chroma(J, m.chroma_opp(J))
    err = float(np.max(np.abs(back - J)))
    rows.append({'shape': [h, w], 'chroma_roundtrip_max_error': err})
    assert err < 1e-8, ('apply_chroma is not the inverse of chroma_opp', err)
print(json.dumps({'checks': rows, 'status': 'passed'}, indent=2))
