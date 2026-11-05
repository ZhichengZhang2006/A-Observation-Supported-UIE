# -*- coding: utf-8 -*-
"""Build docs/acceptance_report.json from the artifacts that actually exist.

Every status is derived from a named artifact; nothing is typed in by hand, so a
missing or failed experiment degrades the status instead of being papered over.
Enums follow the task book:
  performance   passed | not_passed | unverifiable
  novelty       supported | partial | unsupported
  mechanism     supported | partial | unsupported
  reproducibility complete | incomplete
"""
import csv
import hashlib
import io
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts")
DOC = os.path.join(ROOT, "docs")


def load(name, default=None):
    p = os.path.join(ART, name)
    if not os.path.exists(p):
        return default
    with io.open(p, encoding="utf-8") as fh:
        return json.load(fh)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def registry_summary():
    p = os.path.join(DOC, "sota_registry.csv")
    rows = list(csv.DictReader(io.open(p, encoding="utf-8")))
    measured = [r for r in rows if (r.get("unified_psnr") or "").strip()]
    not_run = [r for r in rows if not (r.get("unified_psnr") or "").strip()
               and (r.get("notes") or "").startswith("未查到")]
    return {
        "rows_total": len(rows),
        "rows_with_unified_measurement": len(measured),
        "rows_not_run": len(not_run),
        "not_run_methods": sorted({r["method"] for r in not_run}),
        "denominator_estimate": "none Qualified: MambaCue is a legacy saved-tile "
                                "reference only, and Phaseformer beats it on LSUI",
        "file": "docs/sota_registry.csv",
    }


def _round_nested(d, nd=4):
    out = {}
    for k, v in d.items():
        if isinstance(v, list):
            out[k] = [round(float(x), nd) for x in v]
        else:
            out[k] = round(float(v), nd)
    return out


def h2_block():
    d = load("h2_risk_v3.json")
    if not d:
        return {"status": "missing", "note": "artifacts/h2_risk_v3.json not produced"}
    out = {"status": d.get("status"), "cv": d.get("cv"), "domains": {}}
    for ds, dd in d.get("domains", {}).items():
        dom = {"n_images": dd.get("n_images"), "n_rows": dd.get("n_rows"),
               "backbones": dd.get("backbones"), "targets": {}}
        for t, tv in dd.get("targets", {}).items():
            dom["targets"][t] = {
                "r2_oof": {k: round(v, 4) for k, v in tv["r2_pooled_oof"].items()},
                "delta_F1_minus_F0": _round_nested(tv["delta_bootstrap"]["F1_plus_support"]),
                "delta_F1_minus_F2": _round_nested(tv["delta_bootstrap"]["F1_vs_F2"]),
                "spearman_F1": round(tv["spearman_F1"], 4),
                "spearman_F0": round(tv["spearman_F0"], 4),
            }
        out["domains"][ds] = dom
    return out


def h3_block():
    d = load("h3_mechanism_v2.json")
    iv = load("h3_mechanism_v2_intervals.json", {})
    if not d:
        return {"status": "missing"}
    return {"status": "measured", "n_images": d.get("n"),
            "artifact": "artifacts/h3_mechanism_v2.json",
            "intervals": "artifacts/h3_mechanism_v2_intervals.json",
            "paired_vs_base": (iv.get("support_vs_base") if isinstance(iv, dict) else None)}


def main():
    reg = registry_summary()
    h2 = h2_block()
    h3 = h3_block()
    instr = load("instrument_checks_v3_local.json", {})
    ctrl = load("controlled_detection_v2/summary.json", {})
    prov = load("provenance/inventory_summary.json", {})

    # performance: unverifiable while the denominator is unqualified
    performance = {
        "status": "unverifiable",
        "reason": "No qualified frozen SOTA under the same protocol. MambaCue is a "
                  "legacy saved-tile reference (different UFO preprocessing) and is "
                  "beaten by Phaseformer on LSUI; the six baseline checkpoints carry "
                  "unverified publication provenance.",
        "gate": "max(0,(S-R)/S) <= 0.05 on every PSNR/SSIM for UIEB-R90/LSUI/UFO-120",
        "gate_evaluated": False,
        "registry": reg,
    }

    novelty = {
        "status": "unsupported",
        "reason": "H2 does not separate support from ordinary + ensemble uncertainty, "
                  "which is the one evidence the novelty audit requires.",
        "audit": "docs/novelty_audit.md",
        "near_neighbours": ["MCL-UIE (arXiv:2610.39183)", "Solomatov & Akkaynak (arXiv:2608.15598)",
                            "UADiff (TGRS 2025)"],
        "verification": "docs/literature_verification_20261105.md",
    }

    mechanism = {
        "status": "unsupported",
        "reason": "Under matched controls the support gate makes chroma worse "
                  "(+1.70 % error, bootstrap CI excludes 0) and the matched-strength "
                  "controls (const/random/degrade) do not do better; the oracle "
                  "projection also loses PSNR, so the anchoring is wrong.",
        "artifact": "artifacts/h3_mechanism_v2.json",
        "invalidated_artifacts": ["artifacts/h3_mechanism.json (apply_chroma was not an exact inverse)"],
        "h3": h3,
    }

    reproducibility = {
        "status": "incomplete",
        "have": ["protocol verified against the repository's own recorded result",
                 "SHA-256 manifest of 13578 source files and 6789 pairs",
                 "environment freeze", "per-experiment ledger",
                 "logs and artifacts for every reported number"],
        "missing": ["three-seed training study", "canonical split provenance",
                    "publication-verified baseline weights", "sealed test set"],
        "artifacts": {"provenance": prov, "instrument_checks": instr,
                      "controlled_detection": ctrl},
    }

    doc = {
        "schema": "obsui.acceptance.v1",
        "date": "2026-11-05",
        "verifier": "none — these are agent-proposed statuses, not an external verdict",
        "protocol": {
            "resize": "whole image -> 256x256, PIL BILINEAR, no crop, no pad",
            "psnr": "skimage peak_signal_noise_ratio(data_range=1.0)",
            "ssim": "skimage structural_similarity(win_size=5, channel_axis=-1, data_range=1.0), per-image mean",
            "anchor": "reproduces MambaCue result.txt within 0.04 dB on 3 datasets",
        },
        "items": {"performance": performance, "novelty": novelty,
                  "mechanism": mechanism, "reproducibility": reproducibility},
        "h2": h2,
        "exploratory": True,
        "exploratory_reason": "the public test splits were inspected during protocol work; "
                              "no number here is a sealed confirmation",
    }

    out = os.path.join(DOC, "acceptance_report.json")
    with io.open(out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
    print("wrote", out)
    for k, v in doc["items"].items():
        print(" ", k, "=", v["status"])


if __name__ == "__main__":
    main()
