#!/usr/bin/env python
"""Score models on evaluation suites that no run was selected on, with two probes, and record it all as a run.

The suites come from the pinned kev-suites revision: SemIf (authored evidence judgements plus perturbed siblings),
Scienthoon (support tickets with queue, priority and anger questions), the binding diagnostic (policy fields that
must be bound to the right case fields), Kev's public pool restricted to ARC, OpenBookQA and CommonsenseQA (never
in any training set here), transfer-r3, and the two development splits Kev's model card reports, used only to
validate the Kev port. Probes: option-order permutation (choice questions with their options reversed; the answer
must not move) and state removal (the state replaced by nothing; how much of the accuracy survives).

Results: runs/<id>-suite-eval/final.json, with per-question predictions per model and suite under predictions/.

    uv run python scripts/eval_suites.py --config configs/suite-eval.json [--limit N] [--dry-run]
"""
import argparse
import copy
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tino.data import SUITES, SUITES_REPO, SUITES_REV, build_rows, load_records, sha256
from tino.format import Encoder
from tino.metrics import fit_temperature, softmax, summarize
from tino.track import REPO_ROOT, Run

# Suites beyond tino.data.SUITES: split -> (file, manifest) in the same dataset revision.
EXTRA = {
    "decision-v4": {s: (f"v4/decision-v4/{s}.jsonl", "v4/decision-v4/manifest.json") for s in ("development", "test", "calibration")},
    "transfer-r3": {s: (f"round3/transfer-r3/{s}.jsonl", "round3/transfer-r3/manifest.json") for s in ("development", "test", "calibration")},
    "public-pool-v6": {s: (f"public-pool-v6/{s}.jsonl", "public-pool-v6/manifest.json") for s in ("development", "test", "calibration")},
    "semif-v1": {"development": ("external/semif-v1/development.jsonl", "external/semif-v1/manifest.json")},
    "scienthoon-v1": {"development": ("external/scienthoon-v1/development.jsonl", "external/scienthoon-v1/manifest.json")},
    "binding-v1": {"all": ("diagnostics/binding-v1.jsonl", "diagnostics/binding-v1.manifest.json")},
}


def load_any(suite, split):
    """Records and provenance for a tino suite or one of the extra suites; the manifest checksum is verified when present."""
    if suite in SUITES:
        return load_records(suite, split)
    file, man = EXTRA[suite][split]
    path = hf_hub_download(SUITES_REPO, file, repo_type="dataset", revision=SUITES_REV)
    manifest = json.load(open(hf_hub_download(SUITES_REPO, man, repo_type="dataset", revision=SUITES_REV)))
    digest = sha256(path)
    expected = ((manifest.get("files") or {}).get(os.path.basename(file)) or {}).get("sha256") or manifest.get("sha256")
    if expected and expected != digest:
        raise SystemExit(f"{file}: sha256 {digest} does not match the manifest's {expected}")
    recs = [json.loads(line) for line in open(path)]
    return recs, {"repo": SUITES_REPO, "revision": SUITES_REV, "file": file, "sha256": digest,
                  "manifest_checked": bool(expected), "records": len(recs)}


def spec_of(s):
    spec = dict(s) if isinstance(s, dict) else {"suite": s}
    spec.setdefault("name", spec["suite"])
    return spec


def resolve(m):
    """-> (local dir or Hub repo, revision) for a model entry: {"name", "run"} or {"name", "path"}."""
    if "path" in m:
        return os.path.join(REPO_ROOT, m["path"]), None
    run_dir = m["run"]
    if "*" in run_dir:  # the newest run with that name, so a config can be written before the run exists
        matches = sorted(glob.glob(os.path.join(REPO_ROOT, run_dir)))
        if not matches:
            raise FileNotFoundError(run_dir)
        run_dir = matches[-1]
    ck = json.load(open(os.path.join(REPO_ROOT, run_dir, "checkpoint.json")))
    local = os.path.join(REPO_ROOT, ck["local_dir"])
    if os.path.isdir(local):
        return local, None
    return ck["published"]["repo"], ck["published"]["revision"]


def permuted(recs):
    """Choice questions with their options in reverse order; labels are keys, so they follow their option."""
    out = copy.deepcopy(recs)
    for r in out:
        for q in r["questions"].values():
            if q["type"] == "choice" and isinstance(q.get("criteria"), dict) and len(q["criteria"]) > 1:
                q["criteria"] = dict(reversed(list(q["criteria"].items())))
    return out


def without_state(recs):
    return [dict(r, state="") for r in recs]


def rows_for(enc, recs, spec, limit):
    suite, split = spec["suite"].split("/")
    recs = recs[:limit] if limit else recs
    rows, skipped = build_rows(enc, recs, suite)
    if spec.get("sources"):
        rows = [r for r in rows if r["src"] in spec["sources"]]
    return rows, skipped


def semif_groups(recs, rows, preds):
    """Perturbation groups: a clean record and its perturbed siblings share _meta.group_id."""
    groups = defaultdict(dict)
    for r, p in zip(rows, preds):
        meta = recs[r["rec_idx"]].get("_meta") or {}
        ok = int(np.argmax(p["logits"]) == p["label"])
        groups[meta.get("group_id")].setdefault(meta.get("variant", "clean"), []).append(ok)
    full = [g for g in groups.values() if "clean" in g and len(g) > 1]
    if not full:
        return {"groups": 0}
    clean = [all(g["clean"]) for g in full]
    pert = [all(ok for v, oks in g.items() if v != "clean" for ok in oks) for g in full]
    return {"groups": len(full), "clean_correct": float(np.mean(clean)),
            "perturbed_all_correct": float(np.mean(pert)),
            "both_correct": float(np.mean([c and p for c, p in zip(clean, pert)])),
            "flip_clean_right_perturbed_wrong": float(np.mean([c and not p for c, p in zip(clean, pert)]))}


def permutation_metrics(rows_a, preds_a, rows_b, preds_b):
    """Agreement between the original and the reversed option order, mapped back through the option keys."""
    b = {(r["rec_idx"], r["qid"]): (r, p) for r, p in zip(rows_b, preds_b)}
    agree, both, tvd, by_src = [], [], [], defaultdict(list)
    for r, p in zip(rows_a, preds_a):
        if r["qtype"] != "choice" or (r["rec_idx"], r["qid"]) not in b:
            continue
        rb, pb = b[(r["rec_idx"], r["qid"])]
        pa_ = softmax(p["logits"])
        pb_ = softmax(pb["logits"])
        pb_mapped = np.array([pb_[rb["keys"].index(k)] for k in r["keys"]])
        same = int(np.argmax(pa_) == np.argmax(pb_mapped))
        agree.append(same)
        both.append(int(np.argmax(pa_) == r["label"] and np.argmax(pb_mapped) == r["label"]))
        tvd.append(0.5 * float(np.abs(pa_ - pb_mapped).sum()))
        by_src[r["src"] or "?"].append(same)
    if not agree:
        return {"n": 0}
    return {"n": len(agree), "same_answer": float(np.mean(agree)), "both_correct": float(np.mean(both)),
            "mean_total_variation": float(np.mean(tvd)),
            "by_source": {k: {"n": len(v), "same_answer": float(np.mean(v))} for k, v in sorted(by_src.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0, help="records per suite (smoke test); 0 = all")
    ap.add_argument("--dry-run", action="store_true", help="load suites and build rows only; no model, no run")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    limit = args.limit

    if args.dry_run:
        from transformers import AutoTokenizer
        src, _ = resolve(cfg["models"][0])
        enc = Encoder(AutoTokenizer.from_pretrained(src), 384, 1536)
        for m in cfg["models"]:
            print("[model]", m["name"], "->", resolve(m))
        for s in cfg["suites"] + [x for v in cfg.get("probes", {}).values() for x in v]:
            spec = spec_of(s)
            recs, prov = load_any(*spec["suite"].split("/"))
            rows, skipped = rows_for(enc, recs, spec, limit)
            print(f"[suite] {spec['name']}: {prov['records']} records, {len(rows)} rows, skipped {skipped}, "
                  f"checksum {'ok' if prov['manifest_checked'] else 'not in manifest'}, "
                  f"types {dict(__import__('collections').Counter(r['qtype'] for r in rows))}")
        return

    import mlx.core as mx  # noqa: F401  (MLX only past this point)
    from train import write_predictions
    from tino.evaluate import predict
    from tino.guard import ResourceGuard
    from tino.model import load_checkpoint

    run = Run(cfg["name"] + ("-smoke" if limit else ""), cfg | {"limit": limit}, allow_dirty=bool(limit))
    print(f"[run] {run.id}", flush=True)
    guard = ResourceGuard(**cfg.get("guard", {}), log=lambda m: print(m, flush=True))
    budget = cfg.get("token_budget", 4096)
    cal_suite, cal_split = cfg.get("calibration", ["decision-v7", "calibration"])
    results = {"run_id": run.id, "limit": limit, "models": {}}
    for m in cfg["models"]:
        try:
            src, rev = resolve(m)
        except FileNotFoundError as e:
            print(f"[skip] {m['name']}: no run matches {e}", flush=True)
            results["models"][m["name"]] = {"skipped": f"no run matches {e}"}
            continue
        model, tok, meta = load_checkpoint(src, rev)
        model.eval()
        enc = Encoder(tok, meta.get("max_state", 384), meta.get("max_row", 1536))
        T = float(meta.get("temperature", 1.0))
        res = {"source": src if rev is None else f"{src}@{rev}", "temperature": T, "suites": {}, "probes": {}}
        if m.get("fit_temperature"):
            recs, _ = load_any(cal_suite, cal_split)
            rows, _ = rows_for(enc, recs, {"suite": f"{cal_suite}/{cal_split}"}, limit)
            preds, _ = predict(model, rows, enc.pad_id, budget, guard)
            T = fit_temperature(preds)
            res["temperature"] = T
            res["temperature_fitted_here"] = True
        for s in cfg["suites"]:
            spec = spec_of(s)
            if spec.get("models") and m["name"] not in spec["models"]:
                continue
            recs, prov = load_any(*spec["suite"].split("/"))
            rows, skipped = rows_for(enc, recs, spec, limit)
            preds, spq = predict(model, rows, enc.pad_id, budget, guard)
            entry = {"n": len(rows), "data": prov, "skipped": skipped, "sec_per_question": spq,
                     "raw": summarize(preds), "calibrated": summarize(preds, T)}
            if spec["suite"].startswith("semif"):
                entry["groups"] = semif_groups(recs, rows, preds)
            write_predictions(run.dir, f"{m['name']}/{spec['name']}", rows, preds)
            res["suites"][spec["name"]] = entry
            print(f"[eval] {m['name']} {spec['name']}: n {len(rows)} acc {entry['raw']['accuracy']:.4f} "
                  f"ece {entry['raw']['ece']:.3f} (cal {entry['calibrated']['ece']:.3f})", flush=True)
        for s in cfg.get("probes", {}).get("permutation", []):
            spec = spec_of(s)
            recs, _ = load_any(*spec["suite"].split("/"))
            rows_a, _ = rows_for(enc, recs, spec, limit)
            rows_b, _ = rows_for(enc, permuted(recs), spec, limit)
            preds_a, _ = predict(model, rows_a, enc.pad_id, budget, guard)
            preds_b, _ = predict(model, rows_b, enc.pad_id, budget, guard)
            pm = permutation_metrics(rows_a, preds_a, rows_b, preds_b)
            res["probes"].setdefault("permutation", {})[spec["name"]] = pm
            print(f"[probe] {m['name']} permutation {spec['name']}: same answer {pm.get('same_answer', float('nan')):.3f} "
                  f"on {pm.get('n')} choice questions", flush=True)
        for s in cfg.get("probes", {}).get("no_state", []):
            spec = spec_of(s)
            recs, _ = load_any(*spec["suite"].split("/"))
            rows_a, _ = rows_for(enc, recs, spec, limit)
            rows_b, _ = rows_for(enc, without_state(recs), spec, limit)
            preds_a, _ = predict(model, rows_a, enc.pad_id, budget, guard)
            preds_b, _ = predict(model, rows_b, enc.pad_id, budget, guard)
            full, none = summarize(preds_a), summarize(preds_b)
            res["probes"].setdefault("no_state", {})[spec["name"]] = {
                "n": len(rows_a), "accuracy_with_state": full["accuracy"], "accuracy_without_state": none["accuracy"],
                "by_source": {k: {"with": full["by_source"][k]["accuracy"], "without": none["by_source"].get(k, {}).get("accuracy")}
                              for k in full["by_source"]}}
            print(f"[probe] {m['name']} no_state {spec['name']}: {full['accuracy']:.3f} -> {none['accuracy']:.3f}", flush=True)
        results["models"][m["name"]] = res
        run.write("final.json", results)
        run.sync(f"suite eval: {m['name']} done")
        del model
    run.write("final.json", results)
    run.sync("suite eval: all models done")
    print(f"[done] {run.id}", flush=True)


if __name__ == "__main__":
    main()
