"""Check that adding the extra models leaves every original result unchanged.

For each result file, every record of the original run (lists of dicts, compared in order) must reappear
in the extra-model run with the same numeric values. Records that only exist in the extra-model run (liquid, pplx) are
ignored. Usage (from the repository root): python extra_models/compare_original.py
"""
import json, math, re, sys

PAIRS = [(f"results/{f}", f"results/extra_models/{f}") for f in
         ["r02_core.json", "r03_policies_groups.json", "r04_calibration_size.json", "r05_sensitivity.json", "r06_revision.json",
          "r07_empty_policy_check.json", "r08_g5_costs.json", "r09_scorer_table.json"]]
PAIRS.append(("results/confirmatory/confirmatory_results.json", "results/extra_models/confirmatory/confirmatory_results.json"))
SKIP = {"script_sha256", "inputs", "run_at", "data", "C1_drift", "not_run", "predictions"}   # run metadata; predictions compared separately


def num(v): return isinstance(v, (int, float)) and not isinstance(v, bool)
def same(a, b): return (a is None and b is None) or (num(a) and num(b) and (math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) or (math.isnan(a) and math.isnan(b)))) or a == b
NEW = re.compile(r"d1-ultra|pplx|liquid")
def is_new(r): return any(isinstance(v, str) and NEW.search(v) for v in r.values())


def walk(a, b, path, out):
    if isinstance(a, dict):
        if not isinstance(b, dict): out.append(f"{path}: type differs"); return
        for k, v in a.items():
            if path == "" and k in SKIP: continue
            if k not in b: out.append(f"{path}/{k}: missing"); continue
            walk(v, b[k], f"{path}/{k}", out)
    elif isinstance(a, list) and a and all(isinstance(x, dict) for x in a):
        b = [r for r in b if not is_new(r)]                   # drop the extra models' records, keep the order of the rest
        if len(a) != len(b): out.append(f"{path}: {len(a)} records -> {len(b)}"); return
        for i, (r, q) in enumerate(zip(a, b)):
            for k, v in r.items():                            # only the original columns (wide tables gain extra-model columns)
                if not same(v, q.get(k)): out.append(f"{path}[{i}] {k}: {v} -> {q.get(k)}")
    elif isinstance(a, list):
        if len(a) != len(b) or not all(same(x, y) for x, y in zip(a, b)): out.append(f"{path}: {a} -> {b}")
    elif not same(a, b): out.append(f"{path}: {a} -> {b}")


bad = 0
for old, new in PAIRS:
    A, B = json.load(open(old)), json.load(open(new)); diffs = []
    walk(A, B, "", diffs)
    if old.endswith("confirmatory_results.json"):
        pb = {p["id"]: p for p in B["predictions"]}
        for p in A["predictions"]:
            q = pb.get(p["id"])
            if q is None or q["observed"] != p["observed"] or q["verdict"] != p["verdict"]:
                diffs.append(f"prediction {p['id']}: {p['observed']} / {p['verdict']} -> {q and q['observed']} / {q and q['verdict']}")
    print(f"{'OK     ' if not diffs else 'DIFFERS'} {old}" + (f" ({len(diffs)} differences)" if diffs else ""))
    for d in diffs[:12]: print("   ", d)
    bad += bool(diffs)
sys.exit(1 if bad else 0)
