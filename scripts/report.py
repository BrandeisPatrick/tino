"""Collect run results and compare runs question by question.

    uv run python scripts/report.py                       # table of every finished run
    uv run python scripts/report.py --pair A B --split transfer-v4/test
A and B are run-id substrings. The paired comparison uses the per-question predictions both runs saved: exact
McNemar test on the questions where exactly one run is right, and a bootstrap 95% interval for the accuracy gap.
"""
import argparse
import glob
import gzip
import json
import math
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def runs():
    out = []
    for d in sorted(glob.glob(os.path.join(ROOT, "runs", "*"))):
        f = os.path.join(d, "final.json")
        if os.path.exists(f) and os.path.exists(os.path.join(d, "config.json")) and "smoke" not in d:
            out.append((os.path.basename(d), json.load(open(os.path.join(d, "config.json"))), json.load(open(f))))
    return out


def find(sub):
    hits = [r for r in runs() if sub in r[0]]
    if len(hits) != 1:
        raise SystemExit(f"{sub!r} matches {len(hits)} runs: {[h[0] for h in hits]}")
    return hits[0]


def preds(run_id, split):
    path = os.path.join(ROOT, "runs", run_id, "predictions", split.replace("/", "--") + ".jsonl.gz")
    with gzip.open(path, "rt") as f:
        return {(r["rec_idx"], r["qid"]): r for r in map(json.loads, f)}


def correct(r):
    z = r["logits"]
    return int(max(range(len(z)), key=z.__getitem__) == r["label"])


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def pair(a, b, split, boots=10000):
    pa, pb = preds(a, split), preds(b, split)
    keys = sorted(set(pa) & set(pb))
    ca = [correct(pa[k]) for k in keys]
    cb = [correct(pb[k]) for k in keys]
    n = len(keys)
    only_a = sum(1 for x, y in zip(ca, cb) if x and not y)
    only_b = sum(1 for x, y in zip(ca, cb) if y and not x)
    diffs = [x - y for x, y in zip(ca, cb)]
    rng = random.Random(0)
    bs = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(boots))
    return {"split": split, "n": n, "acc_a": sum(ca) / n, "acc_b": sum(cb) / n, "gap": (sum(ca) - sum(cb)) / n,
            "ci95": [bs[int(0.025 * boots)], bs[int(0.975 * boots)]], "only_a_right": only_a, "only_b_right": only_b,
            "mcnemar_p": mcnemar(only_a, only_b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2)
    ap.add_argument("--split", default="transfer-v4/test")
    args = ap.parse_args()
    if args.pair:
        a, b = find(args.pair[0])[0], find(args.pair[1])[0]
        r = pair(a, b, args.split)
        print(json.dumps({"a": a, "b": b} | r, indent=1))
        return
    for rid, cfg, fin in runs():
        if "evals" not in fin:
            continue
        cells = []
        for split in ("decision-v7/development", "decision-v7/test", "transfer-v4/development", "transfer-v4/test", "transfer-v9/test"):
            e = fin["evals"].get(split)
            cells.append("–" if not e else f"{e['raw']['accuracy']:.3f}")
        print(f"| {rid} | {' | '.join(cells)} |")


if __name__ == "__main__":
    main()
