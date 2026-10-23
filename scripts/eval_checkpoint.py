"""Re-score a finished run's checkpoint on given splits and save per-question predictions into that run's directory
(for paired comparisons between runs). Does not touch the run's final.json.

    uv run python scripts/eval_checkpoint.py --run runs/<id> --splits decision-v7/test transfer-v4/test transfer-v9/test
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import write_predictions
from tino.data import build_rows, load_records
from tino.evaluate import predict
from tino.format import Encoder
from tino.guard import ResourceGuard
from tino.metrics import summarize
from tino.model import load_checkpoint
from tino.track import REPO_ROOT, git


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    args = ap.parse_args()
    run_dir = os.path.join(REPO_ROOT, args.run)
    ck = json.load(open(os.path.join(run_dir, "checkpoint.json")))
    local = os.path.join(REPO_ROOT, ck["local_dir"])
    src, rev = (local, None) if os.path.isdir(local) else (ck["published"]["repo"], ck["published"]["revision"])
    guard = ResourceGuard(mlx_cap_gb=10.0)
    model, tok, meta = load_checkpoint(src, rev)
    model.eval()
    enc = Encoder(tok, meta.get("max_state", 384), meta.get("max_row", 1536))
    final = json.load(open(os.path.join(run_dir, "final.json")))
    for name in args.splits:
        suite, split = name.split("/")
        recs, _ = load_records(suite, split)
        rows, _ = build_rows(enc, recs, suite)
        preds, _ = predict(model, rows, enc.pad_id, 4096, guard)
        write_predictions(run_dir, name, rows, preds)
        acc = summarize(preds)["accuracy"]
        ref = final["evals"].get(name, {}).get("raw", {}).get("accuracy")
        print(f"[eval] {args.run} {name}: acc {acc:.4f} (final.json {ref})", flush=True)
        if ref is not None and abs(acc - ref) > 1e-9:
            raise SystemExit(f"re-scored accuracy {acc} differs from the run's recorded {ref}")
    rel = os.path.relpath(run_dir, REPO_ROOT)
    git("add", rel)
    git("commit", "-m", f"run {os.path.basename(run_dir)}: per-question predictions for {', '.join(args.splits)}", "--", rel)
    git("push", "-q", "origin", "HEAD")


if __name__ == "__main__":
    main()
