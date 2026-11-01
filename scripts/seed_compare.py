"""Compare recipes across seeds, question by question (interim Phase 2 analysis; docs/PHASE2.md).

For each recipe, every finished full-data run with that name (seed suffixes -s1, -s2 folded together) is loaded from
its saved predictions. A question's score for a recipe is its correctness averaged over that recipe's seeds. Each
comparison is a paired sign-flip permutation test on those per-question differences, with a bootstrap 95% interval,
Holm's correction within each research question (the Phase 2 rule) and across all comparisons (stricter), and
whether every seed-matched pair (seed k of A against seed k of B) agrees in sign. Phase 2 claims a difference when the
corrected interval excludes zero and every seed agrees; otherwise it is a lean. Seeds are averaged, not modelled.

    uv run python scripts/seed_compare.py            # prints tables, writes docs/seed_compare.json
"""
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ["decision-v7/test", "transfer-v4/test", "transfer-v9/test"]
RECIPES = {"full fine-tune": "tino-finetuned", "adapter": "tino-adapter", "adapter, post-trained start": "tino-adapter-post",
           "teacher 0.25": "tino-distil-a025", "teacher 0.5": "tino-distil", "teacher 0.75": "tino-distil-a075",
           "teacher only": "tino-distil-a1", "teacher 0.5, unsoftened": "tino-distil-raw"}
COMPARISONS = [("adapter", "full fine-tune", "RQ2"), ("teacher 0.5", "adapter", "RQ4"), ("teacher only", "adapter", "RQ4"),
               ("teacher 0.25", "adapter", "RQ4"), ("teacher 0.75", "adapter", "RQ4"),
               ("teacher 0.5, unsoftened", "teacher 0.5", "H4"), ("adapter, post-trained start", "adapter", "RQ3b")]


def finished_runs(base):
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", "*"))):
        rid = os.path.basename(d)
        name = rid[16:]
        if re.fullmatch(re.escape(base) + r"(-s\d+)?", name) and os.path.exists(os.path.join(d, "final.json")) \
                and not os.path.exists(os.path.join(d, "aborted.json")):
            out.append(rid)
    return out


def correctness(rid, split):
    with gzip.open(os.path.join(ROOT, "runs", rid, "predictions", split.replace("/", "--") + ".jsonl.gz"), "rt") as f:
        rows = [json.loads(l) for l in f]
    return {(r["rec_idx"], r["qid"]): int(int(np.argmax(r["logits"])) == r["label"]) for r in rows}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def main(perms=20000, boots=5000, seed=0):
    rng = np.random.default_rng(seed)
    data = {}
    for label, base in RECIPES.items():
        rids = finished_runs(base)
        data[label] = {"runs": rids, "splits": {sp: [correctness(r, sp) for r in rids] for sp in SPLITS}}
    summary = {}
    for label, d in data.items():
        summary[label] = {"seeds": len(d["runs"]), "runs": d["runs"]}
        for sp in SPLITS:
            accs = [np.mean(list(c.values())) for c in d["splits"][sp]]
            summary[label][sp] = {"mean": float(np.mean(accs)), "sd": float(np.std(accs, ddof=1)) if len(accs) > 1 else None,
                                  "per_seed": [round(float(a), 4) for a in accs]}
    tests = []
    for a, b, q in COMPARISONS:
        for sp in SPLITS:
            ca, cb = data[a]["splits"][sp], data[b]["splits"][sp]
            keys = sorted(set.intersection(*[set(c) for c in ca + cb]))
            da = np.mean([[c[k] for k in keys] for c in ca], axis=0)
            db = np.mean([[c[k] for k in keys] for c in cb], axis=0)
            diff = da - db
            obs = diff.mean()
            hits = 0
            for s in range(0, perms, 2000):
                signs = rng.choice([-1.0, 1.0], size=(min(2000, perms - s), len(diff)))
                hits += int((np.abs((signs * diff).mean(axis=1)) >= abs(obs) - 1e-12).sum())
            p = (hits + 1) / (perms + 1)
            bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(boots)]
            matched = [float(np.mean([x[k] for k in keys]) - np.mean([y[k] for k in keys])) for x, y in zip(ca, cb)]
            tests.append({"a": a, "b": b, "question": q, "split": sp, "n": len(keys), "seeds": [len(ca), len(cb)],
                          "diff": float(obs), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "p": p,
                          "seed_matched_diffs": [round(m, 4) for m in matched],
                          "seeds_agree": bool(all(np.sign(m) == np.sign(obs) and m != 0 for m in matched))})
    for t, pa in zip(tests, holm([t["p"] for t in tests])):
        t["p_holm_all"] = pa
    for q in {t["question"] for t in tests}:
        grp = [t for t in tests if t["question"] == q]
        for t, pa in zip(grp, holm([t["p"] for t in grp])):
            t["p_holm"] = pa
    for t in tests:
        t["verdict"] = "claim" if t["p_holm"] < 0.05 and t["seeds_agree"] else ("lean" if t["p"] < 0.05 else "no difference shown")
    json.dump({"recipes": summary, "tests": tests, "method": __doc__.strip().split("\n\n")[1]},
              open(os.path.join(ROOT, "docs", "seed_compare.json"), "w"), indent=1)
    print(f"{'recipe':<30} seeds " + " ".join(f"{sp:>22}" for sp in SPLITS))
    for label, s in summary.items():
        print(f"{label:<30} {s['seeds']:>5} " + " ".join(
            f"{s[sp]['mean']:.3f} ± {s[sp]['sd']:.3f}".rjust(22) if s[sp]["sd"] is not None else f"{s[sp]['mean']:.3f}".rjust(22) for sp in SPLITS))
    print()
    for t in tests:
        print(f"{t['question']:<5} {t['a']:<28} vs {t['b']:<16} {t['split']:<18} {t['diff']*100:+5.1f} pts "
              f"[{t['ci95'][0]*100:+5.1f}, {t['ci95'][1]*100:+5.1f}]  p {t['p']:.4f}  Holm(RQ) {t['p_holm']:.4f}  Holm(all) {t['p_holm_all']:.4f}  "
              f"seeds {' '.join(f'{m*100:+.1f}' for m in t['seed_matched_diffs'])}  {t['verdict']}")


if __name__ == "__main__":
    main()
