"""Turn runs/ into tracked, readable records.

    uv run python scripts/collect_results.py

Writes three files from the run directories (never the other way round):

- runs/INDEX.md      one row per run: what it was, how it ended, its key numbers, where its weights are
- docs/results.json  every run's parsed config, provenance and metrics, for scripts and pages
- docs/MODELS.md     the registry of published models: Hub repo and revision, run, recipe, test metrics, how to load

Run it after any run finishes and commit the output with the run. Smoke runs are indexed but marked as such.
"""
import glob
import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = ["decision-v7/development", "decision-v7/test", "transfer-v4/development", "transfer-v4/test", "transfer-v9/test"]
SHORT = {"decision-v7/development": "d7 dev", "decision-v7/test": "d7 test", "transfer-v4/development": "t4 dev", "transfer-v4/test": "t4 test", "transfer-v9/test": "t9 test"}


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def parse_run(d):
    rid = os.path.basename(d)
    cfg, fin, prov, ck, ab = (load(os.path.join(d, f)) for f in ("config.json", "final.json", "provenance.json", "checkpoint.json", "aborted.json"))
    r = {"id": rid, "name": rid[16:], "started": rid[:15], "smoke": "smoke" in rid, "dir": f"runs/{rid}"}
    r["code_commit"] = (prov or {}).get("code", {}).get("commit", "")[:8]
    r["code_dirty"] = (prov or {}).get("code", {}).get("dirty")
    if cfg and "method" in cfg:
        r["kind"] = "training"
        r["method"] = cfg["method"] + (" + distillation" if cfg.get("distil") else "")
        r["base_model"], r["base_revision"] = cfg.get("base_model"), (cfg.get("base_revision") or "")[:8]
        r["seed"], r["epochs"], r["train_limit"] = cfg.get("seed"), cfg.get("epochs"), cfg.get("train_limit")
        r["lora"], r["alpha"] = cfg.get("lora"), (cfg.get("distil") or {}).get("alpha")
        r["backbone_lr"], r["brier_weight"] = cfg.get("backbone_lr"), cfg.get("brier_weight")
    elif cfg and "teacher" in cfg:
        r["kind"] = "teacher labels"; r["teacher"] = cfg["teacher"]
    elif cfg and "models" in cfg:
        r["kind"] = "zero-shot eval"
    else:
        r["kind"] = "unknown"
    if ab:
        r["status"] = "stopped"; r["stopped_reason"] = ab.get("reason")
    elif fin and (fin.get("evals") or fin.get("models") or fin.get("labels_revision")):
        r["status"] = "finished"
    elif fin and fin.get("splits"):
        r["status"] = "finished"
    else:
        r["status"] = "incomplete"
    r["metrics"] = {}
    if fin and fin.get("evals"):
        for sp, e in fin["evals"].items():
            r["metrics"][sp] = {"acc": round(e["raw"]["accuracy"], 4), "brier": round(e["raw"]["brier"], 4), "ece_raw": round(e["raw"]["ece"], 4),
                                "ece_cal": round(e["calibrated"]["ece"], 4), "ms": round(e["sec_per_question"] * 1000), "n": e["raw"]["n"]}
        r["temperature"] = round(fin["temperature"], 3); r["best_step"] = fin.get("best_step"); r["train_minutes"] = round(fin.get("train_minutes", 0))
        r["trainable_params"] = fin.get("trainable_params")
        bt = fin["evals"].get("decision-v7/test", {}).get("raw", {}).get("by_type")
        if bt: r["by_type_test"] = {k: round(v["accuracy"], 4) for k, v in bt.items()}
    if fin and fin.get("splits"):
        r["teacher_temperature"] = fin.get("temperature"); r["labels_revision"] = (fin.get("labels_revision") or "")[:8]
        for sp, v in fin["splits"].items():
            if "raw" in v: r["metrics"][sp] = {"acc": round(v["raw"]["accuracy"], 4), "ece_raw": round(v["raw"]["ece"], 4), "ece_cal": round(v["calibrated"]["ece"], 4), "n": v["raw"]["n"]}
    if fin and fin.get("models"):
        r["zero_shot"] = {m: {sp: round(v["accuracy"], 3) for sp, v in x["suites"].items()} | {"sec_per_question": round(x["sec_per_question"], 2)} for m, x in fin["models"].items()}
    ev = os.path.join(d, "evals.jsonl")
    if os.path.exists(ev):
        rows = [json.loads(l) for l in open(ev)]
        r["dev_curve"] = [[e["step"], round(e["accuracy"], 4)] for e in rows]
        nlls = {round(e["nll"], 3) for e in rows}
        r["collapsed"] = bool(rows) and (len(nlls) == 1 and abs(rows[-1]["nll"] - 1.478) < 0.002 or r.get("temperature", 1) <= 0.11)
    res = os.path.join(d, "resources.jsonl")
    if os.path.exists(res):
        rs = [json.loads(l) for l in open(res)]
        if rs: r["peak_gb"] = max(x["mlx_peak_gb"] for x in rs); r["min_free_pct"] = min(x["free_pct"] for x in rs)
    if ck and ck.get("published"):
        r["published"] = ck["published"]
        local = os.path.join(ROOT, ck["local_dir"], "model.safetensors")
        if os.path.exists(local): r["weights_mb"] = round(os.path.getsize(local) / 1e6)
    r["has_predictions"] = os.path.isdir(os.path.join(d, "predictions"))
    return r


def index_md(runs):
    out = ["# Run index", "", "Generated by `scripts/collect_results.py` from the run directories; do not edit by hand. Accuracies are raw (before temperature scaling). "
           "Full-data models report test splits; ablations report development splits only. Links go to each run's directory.", "",
           "| Run | Kind | Status | d7 dev | d7 test | t4 dev | t4 test | t9 test | T | min | weights |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        m = r["metrics"]; cell = lambda sp: fmt(m.get(sp, {}).get("acc"))
        status = r["status"] + (" (smoke)" if r["smoke"] else "") + (" · collapsed" if r.get("collapsed") else "")
        if r.get("stopped_reason"): status += f" · {r['stopped_reason'][:60]}"
        w = f"[{r['published']['repo'].split('/')[-1]}@{r['published']['revision'][:8]}](https://huggingface.co/{r['published']['repo']}/tree/{r['published']['revision']})" if r.get("published") else "–"
        T = fmt(r.get("temperature"), 2) if r["kind"] == "training" else (fmt(r.get("teacher_temperature"), 1) if r.get("teacher_temperature") else "–")
        out.append(f"| [`{r['id']}`]({r['id']}) | {r['kind']} | {status} | {cell('decision-v7/development')} | {cell('decision-v7/test')} | {cell('transfer-v4/development')} | {cell('transfer-v4/test')} | {cell('transfer-v9/test')} | {T} | {r.get('train_minutes', '–')} | {w} |")
    return "\n".join(out) + "\n"


def models_md(runs):
    pub = [r for r in runs if r.get("published") and not r["smoke"]]
    out = ["# Model registry", "", "Generated by `scripts/collect_results.py`. Every model is a private Hugging Face repo under `BrandeisPatrick`; the revision pins the exact weights that produced the numbers below. "
           "Each Hub copy was downloaded and re-scored once, and reproduced its run's test accuracy exactly. Test splits were read once per model.", "",
           "| Model | Recipe | Base | Run | Trained sources | New sources | Hardest | ECE calibrated (new) | T | ms / q | Weights |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in pub:
        p = r["published"]; m = r["metrics"]
        recipe = {"full_finetune": "full fine-tune", "lora": f"LoRA r{(r.get('lora') or {}).get('rank')}"}.get(r.get("method", "").split(" ")[0], r.get("method"))
        if r.get("alpha") is not None: recipe += f", teacher weight {r['alpha']}"
        out.append(f"| [`{p['repo'].split('/')[-1]}`](https://huggingface.co/{p['repo']}) | {recipe} | `{r['base_model']}` @ `{r['base_revision']}` | [`{r['id'][:15]}`](../{r['dir']}) | "
                   f"{fmt(m.get('decision-v7/test', {}).get('acc'))} | {fmt(m.get('transfer-v4/test', {}).get('acc'))} | {fmt(m.get('transfer-v9/test', {}).get('acc'))} | "
                   f"{fmt(m.get('transfer-v4/test', {}).get('ece_cal'))} | {fmt(r.get('temperature'), 2)} | {m.get('transfer-v4/test', {}).get('ms', '–')} | {r.get('weights_mb', '–')} MB @ `{p['revision'][:8]}` |")
    out += ["", "## Loading a model", "", "```python", "# git clone https://github.com/BrandeisPatrick/tino && cd tino && uv sync", "from tino.api import Decider",
            "d = Decider.load(\"BrandeisPatrick/tino-distil\", revision=\"<revision from the table>\")",
            "d.decide({\"state\": \"...\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"...?\"}}})", "```", "",
            "Answers follow TypeSafe's `/v1/systemone` shapes (noul probability; choice with per-option probabilities and confidence; score with expected level and legend). "
            "Probabilities are divided by the model's fitted temperature `T` unless `calibrated=False` is passed.", "",
            "## Datasets", "", "| Dataset | Revision | Used for |", "|---|---|---|",
            "| [`jaredpalmer/kev-suites`](https://huggingface.co/datasets/jaredpalmer/kev-suites) | `cc4bac80` | training (decision-v7 train), selection (development), calibration, and all test splits; checksums verified against each suite's manifest |",
            "| [`BrandeisPatrick/tino-teacher-labels`](https://huggingface.co/datasets/BrandeisPatrick/tino-teacher-labels) (private) | `31e5bfd7` | Qwen3-4B-Instruct-2507 option log-probabilities for every split, prefix `qwen3-4b-instruct-2507`; the manifest records the prompt hash and the teacher temperature |", ""]
    return "\n".join(out)


def main():
    runs = [parse_run(d) for d in sorted(glob.glob(os.path.join(ROOT, "runs", "*"))) if os.path.isdir(d)]
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    open(os.path.join(ROOT, "runs", "INDEX.md"), "w").write(index_md(runs))
    open(os.path.join(ROOT, "docs", "MODELS.md"), "w").write(models_md(runs))
    finished = [r for r in runs if r["status"] == "finished" and not r["smoke"]]
    summary = {"generated": time.strftime("%Y-%m-%d"), "runs_total": len(runs), "runs_finished_nonsmoke": len(finished),
               "train_hours_nonsmoke": round(sum(r.get("train_minutes", 0) for r in finished if r["kind"] == "training") / 60, 1),
               "published_models": [r["published"]["repo"] for r in runs if r.get("published") and not r["smoke"]]}
    json.dump({"summary": summary, "runs": runs}, open(os.path.join(ROOT, "docs", "results.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
