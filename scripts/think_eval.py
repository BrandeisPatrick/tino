"""RQ3a: does reasoning first make a better zero-shot decision maker?

Scores the same fixed development subset with an instruct model (answers directly: one forward pass, letter logits)
and a thinking model (generates its reasoning greedily up to --max-think tokens, then the letter logits are read after
</think>). Records accuracy, calibration, per-source results, thinking length and seconds per question.

    uv run python scripts/think_eval.py --config configs/think-eval.json
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from tino import teacher as tm
from tino.data import load_records
from tino.format import label_index, question_options
from tino.guard import ResourceAbort, ResourceGuard
from tino.metrics import summarize
from tino.track import Run


def subset(cfg):
    items = []
    for suite, split, n in cfg["subset"]:
        recs, prov = load_records(suite, split)
        flat = [(i, qid, q) for i, r in enumerate(recs) for qid, q in r["questions"].items()
                if len(question_options(q)[0]) <= len(tm.LETTERS)]
        for i, qid, q in flat[:: max(1, len(flat) // n)][:n]:
            keys, _ = question_options(q)
            items.append({"suite": f"{suite}/{split}", "rec": recs[i], "rec_idx": i, "qid": qid, "q": q, "k": len(keys),
                          "label": label_index(q, keys), "src": q.get("src"), "qtype": q["type"]})
    return items


def run_direct(T, items, guard):
    for x in items:
        x["ids"] = tm.prompt_ids(T, x["rec"], x["q"])
    t0 = time.time()
    logp = tm.score(T, items, 4096, guard)
    dt = time.time() - t0
    return [{"logits": l, "think_tokens": 0} for l in logp], dt / len(items)


def run_thinking(T, items, guard, max_think, log):
    out, t_all = [], time.time()
    sampler = make_sampler(temp=0.0)
    for n, x in enumerate(items):
        guard.check()
        ids = tm.prompt_ids(T, x["rec"], x["q"])
        text, toks, closed = "", 0, False
        for resp in stream_generate(T["model"], T["tok"], ids, max_tokens=max_think, sampler=sampler):
            text += resp.text
            toks += 1
            if "</think>" in text:
                closed = True
                break
        thought = text.split("</think>")[0]
        full = T["tok"].decode(ids) + thought + "\n</think>\n\n"
        full_ids = T["tok"].encode(full, add_special_tokens=False)
        logp = tm.score(T, [{"ids": full_ids, "k": x["k"]}], 8192, guard)[0]
        out.append({"logits": logp, "think_tokens": toks, "closed": closed})
        if (n + 1) % 25 == 0:
            log(f"[think] {n+1}/{len(items)} questions, {(time.time()-t_all)/(n+1):.1f} s/q, "
                f"mean thinking {sum(o['think_tokens'] for o in out)/len(out):.0f} tokens")
        mx.clear_cache()
    return out, (time.time() - t_all) / len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    smoke = args.limit > 0
    run = Run(cfg["name"] + ("-smoke" if smoke else ""), cfg | {"limit": args.limit}, allow_dirty=smoke)
    guard = ResourceGuard(**cfg["guard"], log=lambda m: print(m, flush=True))
    items = subset(cfg)
    if smoke:
        items = items[:: max(1, len(items) // args.limit)][: args.limit]
    print(f"[run] {run.id}: {len(items)} questions", flush=True)
    results = {"questions": len(items), "models": {}}
    try:
        for m in cfg["models"]:
            T = tm.load_teacher(m["repo"], m["revision"])
            if m["mode"] == "direct":
                outs, spq = run_direct(T, items, guard)
            else:
                outs, spq = run_thinking(T, items, guard, cfg["max_think"], lambda s: print(s, flush=True))
            res = {"repo": m["repo"], "revision": m["revision"], "mode": m["mode"], "sec_per_question": spq,
                   "mean_think_tokens": sum(o["think_tokens"] for o in outs) / len(outs),
                   "unclosed_thinking": sum(1 for o in outs if o.get("closed") is False), "suites": {}}
            for suite in sorted({x["suite"] for x in items}):
                preds = [{"logits": o["logits"], "label": x["label"], "src": x["src"], "qtype": x["qtype"]}
                         for o, x in zip(outs, items) if x["suite"] == suite]
                res["suites"][suite] = summarize(preds)
            res["predictions"] = [{"suite": x["suite"], "rec_idx": x["rec_idx"], "qid": x["qid"], "src": x["src"], "label": x["label"],
                                   "logits": [round(v, 4) for v in o["logits"]], "think_tokens": o["think_tokens"]} for o, x in zip(outs, items)]
            results["models"][m["name"]] = res
            for suite, sm in res["suites"].items():
                print(f"[result] {m['name']} {suite}: acc {sm['accuracy']:.4f} ece {sm['ece']:.4f} (n={sm['n']}), "
                      f"{spq:.2f} s/q, thinking {res['mean_think_tokens']:.0f} tokens", flush=True)
            run.write("final.json", results)
            run.sync(f"think eval: {m['name']} done")
            del T
            mx.clear_cache()
    except ResourceAbort as err:
        run.write("aborted.json", {"reason": str(err), "resources": guard.snapshot()})
        run.sync(f"aborted by resource guard: {err}")
        sys.exit(3)
    print(f"[done] {run.id}", flush=True)


if __name__ == "__main__":
    main()
