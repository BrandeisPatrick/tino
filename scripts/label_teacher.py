"""Label kev-suites splits with an autoregressive LLM teacher (tino/teacher.py) and track everything remotely.

Each split's labels go to a private Hugging Face dataset as soon as that split is done (resumable: splits already on
the Hub are skipped). The run directory on GitHub records the teacher's own zero-shot accuracy and calibration, raw and
with a temperature fitted on the calibration split.

    uv run python scripts/label_teacher.py --config configs/teacher-qwen3-4b.json
"""
import argparse
import hashlib
import inspect
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi

from tino import teacher as teacher_mod
from tino.data import load_records
from tino.format import label_index, question_options
from tino.guard import ResourceAbort, ResourceGuard
from tino.metrics import fit_temperature, summarize
from tino.track import REPO_ROOT, Run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, default=0, help="questions per split (smoke test); 0 = all")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()
    cfg = json.load(open(args.config))
    smoke = args.limit > 0
    run = Run(cfg["name"] + ("-smoke" if smoke else ""), cfg | {"limit": args.limit}, allow_dirty=args.allow_dirty or smoke)
    guard = ResourceGuard(**cfg["guard"], log=lambda m: print(m, flush=True))
    api = HfApi()
    repo = cfg["labels_repo"]
    prefix = cfg["labels_prefix"] + ("-smoke" if smoke else "")
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    existing = set(api.list_repo_files(repo, repo_type="dataset"))
    prompt_hash = hashlib.sha256(inspect.getsource(teacher_mod.prompt_ids).encode()).hexdigest()[:12]
    print(f"[run] {run.id} -> {repo}/{prefix} (prompt {prompt_hash})", flush=True)

    T = teacher_mod.load_teacher(cfg["teacher"], cfg["teacher_revision"])
    out_dir = os.path.join(REPO_ROOT, "checkpoints", run.id)
    os.makedirs(out_dir, exist_ok=True)
    summary = {"teacher": cfg["teacher"], "teacher_revision": cfg["teacher_revision"], "prompt_sha": prompt_hash, "splits": {}}
    preds_by_split = {}
    try:
        for suite, split in cfg["splits"]:
            path_in_repo = f"{prefix}/{suite}/{split}.jsonl"
            recs, prov = load_records(suite, split)
            items = []
            for i, rec in enumerate(recs):
                for qid, q in rec["questions"].items():
                    keys, _ = question_options(q)
                    items.append({"rec_idx": i, "qid": qid, "k": len(keys), "label": label_index(q, keys), "src": q.get("src"),
                                  "qtype": q["type"], "ids": teacher_mod.prompt_ids(T, rec, q)})
            if smoke:
                items = items[:: max(1, len(items) // args.limit)][: args.limit]
            if path_in_repo in existing and not smoke:
                print(f"[skip] {path_in_repo} already on the Hub", flush=True)
                local = api.hf_hub_download(repo, path_in_repo, repo_type="dataset")
                rows = [json.loads(l) for l in open(local)]
                summary["splits"][f"{suite}/{split}"] = {"questions": len(rows), "labelled": sum(r["teacher_logp"] is not None for r in rows),
                                                         "path": path_in_repo, "resumed": True, "data": prov}
            else:
                todo = [x for x in items if x["ids"] is not None]
                t0 = time.time()
                logp = teacher_mod.score(T, todo, cfg["token_budget"], guard)
                dt = time.time() - t0
                for x, lp in zip(todo, logp):
                    x["teacher_logp"] = lp
                rows = [{"rec_idx": x["rec_idx"], "qid": x["qid"], "k": x["k"], "label": x["label"], "src": x["src"], "qtype": x["qtype"],
                         "teacher_logp": x.get("teacher_logp")} for x in items]
                local = os.path.join(out_dir, f"{suite}--{split}.jsonl")
                with open(local, "w") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                info = api.upload_file(path_or_fileobj=local, path_in_repo=path_in_repo, repo_id=repo, repo_type="dataset",
                                       commit_message=f"{prefix} {suite}/{split} from run {run.id}")
                print(f"[label] {suite}/{split}: {len(todo)}/{len(items)} questions in {dt/60:.1f} min "
                      f"({dt/max(len(todo),1)*1000:.0f} ms/q) -> {repo}@{info.oid[:8]}", flush=True)
                summary["splits"][f"{suite}/{split}"] = {"questions": len(items), "labelled": len(todo), "minutes": dt / 60,
                                                         "hub_commit": info.oid, "path": path_in_repo, "data": prov}
            preds_by_split[f"{suite}/{split}"] = [{"logits": r["teacher_logp"], "label": r["label"], "src": r["src"], "qtype": r["qtype"]}
                                                  for r in rows if r["teacher_logp"] is not None and r["label"] is not None]
            if f"{suite}/{split}" == "/".join(cfg["calibration"]):
                summary["temperature"] = fit_temperature(preds_by_split[f"{suite}/{split}"])
                print(f"[calib] teacher temperature {summary['temperature']:.3f}", flush=True)
            Tt = summary.get("temperature", 1.0)
            s_raw, s_cal = summarize(preds_by_split[f"{suite}/{split}"]), summarize(preds_by_split[f"{suite}/{split}"], Tt)
            summary["splits"].setdefault(f"{suite}/{split}", {}).update(raw=s_raw, calibrated=s_cal)
            print(f"[teacher] {suite}/{split}: acc {s_raw['accuracy']:.4f} (n={s_raw['n']}) ece raw {s_raw['ece']:.4f} / T={Tt:.2f} {s_cal['ece']:.4f}", flush=True)
            run.write("final.json", summary)
            run.sync(f"teacher labels {suite}/{split}")
    except ResourceAbort as err:
        run.write("aborted.json", {"reason": str(err), "resources": guard.snapshot(), "done": list(summary["splits"])})
        run.sync(f"aborted by resource guard: {err}")
        sys.exit(3)
    # Manifest: which files, which teacher, which prompt, which temperature. The student config pins this dataset revision.
    manifest = {k: summary[k] for k in ("teacher", "teacher_revision", "prompt_sha")} | {
        "teacher_source_model": cfg.get("teacher_source_model"), "temperature": summary.get("temperature"), "run_id": run.id,
        "files": {k: {"path": v.get("path", f"{prefix}/{k}.jsonl"), "questions": v["questions"]} for k, v in summary["splits"].items()},
        "note": "teacher_logp are log-probabilities over option letters A.. (renormalized), raw (T=1); null for >26 options."}
    mpath = os.path.join(out_dir, "manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    info = api.upload_file(path_or_fileobj=mpath, path_in_repo=f"{prefix}/manifest.json", repo_id=repo, repo_type="dataset",
                           commit_message=f"{prefix} manifest from run {run.id}")
    summary["labels_revision"] = info.oid
    run.write("final.json", summary)
    run.sync(f"teacher labels complete, dataset {repo}@{info.oid[:8]}")
    print(f"[done] {run.id} labels at {repo}@{info.oid}", flush=True)


if __name__ == "__main__":
    main()
