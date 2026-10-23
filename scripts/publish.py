"""Upload a finished run's checkpoint to a private Hugging Face repo with a model card built from its results,
then record the Hub revision in the run directory and push that to GitHub.

    uv run python scripts/publish.py --run runs/<run-id> [--repo BrandeisPatrick/tino-finetuned]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import HfApi

from tino.track import REPO_ROOT, git

KEV_06B = {"decision-v7/test": 0.808, "transfer-v4/test": 0.642, "transfer-v4/development": 0.620}
JEV = {"transfer-v4/development": 0.857}


def fmt(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def model_card(cfg, final, run_id, repo):
    rows = []
    for name, r in final["evals"].items():
        raw, cal = r["raw"], r["calibrated"]
        rows.append(f"| {name} | {raw['n']} | {fmt(raw['accuracy'])} | {fmt(raw['brier'])} | {fmt(raw['ece'])} | {fmt(cal['ece'])} | "
                    f"{fmt(raw['confident_error'])} | {fmt(KEV_06B.get(name))} | {fmt(JEV.get(name))} |")
    per_src = final["evals"].get("transfer-v4/test", {}).get("raw", {}).get("by_source", {})
    src_line = ", ".join(f"{k} {v['accuracy']:.2f}" for k, v in per_src.items())
    gh = f"https://github.com/BrandeisPatrick/tino/tree/main/runs/{run_id}"
    return f"""---
license: apache-2.0
base_model: {cfg['base_model']}
pipeline_tag: text-classification
library_name: mlx
tags: [decision-model, calibration, jev, typesafe, mlx, full-finetune]
datasets: [jaredpalmer/kev-suites]
---

# {cfg['name']}

A Jev-style **decision model**: a state plus typed questions (choice, score, yes/no) in, a probability distribution per
question out, in one forward pass with no generated text. This is the **full fine-tune** member of the tino family
(`tino-finetuned`, `tino-adapter`, `tino-distil`): every transformer layer of `{cfg['base_model']}` is trained together
with a pointer head; the token embeddings stay frozen.

Trained on an Apple M4 laptop (16 GB) with MLX. Run record, config, logs, and provenance: [{run_id}]({gh}).

## Results

Accuracy and calibration on frozen [kev-suites](https://huggingface.co/datasets/jaredpalmer/kev-suites) splits
(revision `{final['evals'][next(iter(final['evals']))]['data']['revision'][:8]}`). Test splits were read once. ECE uses 15 bins;
"calibrated" divides logits by T = {final['temperature']:.2f}, fitted on the decision-v7 calibration split. Confident error is the
share of answers with top probability ≥ 0.9 that are wrong. Kev-0.6B is a LoRA adapter on the same base and data
(numbers from its model card); Jev was run by Kev's authors on the development split only.

| Suite / split | Questions | Accuracy | Brier | ECE raw | ECE calibrated | Confident error | Kev-0.6B acc | Jev acc |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

decision-v7 = held-out examples from the ten training datasets ("trained sources"). transfer-v4 = datasets never trained
on ("new sources"). transfer-v9 adds MMLU-Pro, buried evidence, and unknowable records.

Per-source accuracy on transfer-v4 test: {src_line}.

## Training

| | |
|---|---|
| Base | `{cfg['base_model']}` @ `{cfg['base_revision'][:8]}` |
| Trainable | {final['trainable_params']/1e6:.0f}M parameters (all transformer layers + pointer head; embeddings frozen) |
| Data | decision-v7 train: {final.get('train_questions', '15,576')} questions, {cfg['epochs']} epochs |
| Optimizer | Adafactor (backbone, peak lr {cfg['backbone_lr']}), AdamW (head, peak lr {cfg['head_lr']}), warmup {cfg['warmup_frac']:.0%}, cosine to {cfg['min_lr_frac']:.0%} |
| Batching | length-bucketed, ≤ {cfg['token_budget']} padded tokens per step |
| Selection | best development accuracy, step {final['best_step']} |
| Hardware | Apple M4, 16 GB, MLX; {final['train_minutes']:.0f} minutes |

## Usage

```python
# git clone https://github.com/BrandeisPatrick/tino && cd tino && uv sync
from tino.api import Decider
d = Decider.load("{repo}")
d.decide({{
    "state": "Customer asks for a refund on a 40-day-old order. Policy window is 30 days.",
    "questions": {{
        "action": {{"type": "choice", "instructions": "What should support do?",
                    "criteria": {{"refund": None, "escalate": "Needs a manager", "decline": None}}}},
        "urgent": {{"type": "noul", "instructions": "Does this need a reply within an hour?"}},
    }},
}})
```

## Limits

A 0.6B model. Expect much weaker accuracy on sources it never trained on than on trained ones, and treat its
probabilities as advisory out of domain. Input layout, delimiter tokens and pointer head follow
[Kev](https://github.com/jaredpalmer/kev) (Apache-2.0).
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()
    run_dir = os.path.join(REPO_ROOT, args.run) if not os.path.isabs(args.run) else args.run
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    final = json.load(open(os.path.join(run_dir, "final.json")))
    data = json.load(open(os.path.join(run_dir, "data.json")))
    final["train_questions"] = f"{data['train_questions']:,}"
    ck = json.load(open(os.path.join(run_dir, "checkpoint.json")))
    local = os.path.join(REPO_ROOT, ck["local_dir"])
    run_id = os.path.basename(run_dir.rstrip("/"))
    repo = args.repo or f"BrandeisPatrick/{cfg['name']}"
    open(os.path.join(local, "README.md"), "w").write(model_card(cfg, final, run_id, repo))
    json.dump(final, open(os.path.join(local, "results.json"), "w"), indent=1)
    api = HfApi()
    api.create_repo(repo, private=True, exist_ok=True)
    info = api.upload_folder(repo_id=repo, folder_path=local, commit_message=f"{cfg['name']} from run {run_id}")
    rev = info.oid
    ck["published"] = {"repo": repo, "revision": rev, "url": f"https://huggingface.co/{repo}/tree/{rev}"}
    json.dump(ck, open(os.path.join(run_dir, "checkpoint.json"), "w"), indent=1)
    rel = os.path.relpath(run_dir, REPO_ROOT)
    git("add", rel)
    git("commit", "-m", f"run {run_id}: published to {repo}@{rev[:8]}", "--", rel)
    git("push", "-q", "origin", "HEAD")
    print(f"[publish] {repo}@{rev}")


if __name__ == "__main__":
    main()
