"""Train the generative counterpart of a tino decision model (RESEARCH.md, "Later: are decision models more
parameter-efficient?"). Same data, rows, batches, adapter recipe, guard and tracking as scripts/train.py, which is
left untouched; the model writes the chosen option instead of pointing at it (tino/generative.py). The best checkpoint
is chosen by written accuracy on the development split, as decision runs are chosen by their accuracy.

    uv run python scripts/train_gen.py --config configs/tino-gen.json
    uv run python scripts/train_gen.py --config configs/tino-gen.json --smoke   # tiny end-to-end check
"""
import argparse
import gzip
import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from tino import generative as gen
from tino.data import batches, build_rows, load_records
from tino.guard import ResourceAbort, ResourceGuard
from tino.metrics import fit_temperature, summarize
from tino.model import apply_lora, save_checkpoint
from tino.track import REPO_ROOT, Run


def count(params):
    return sum(v.size for _, v in tree_flatten(params))


def write_predictions(run_dir, name, rows, scored, written_ids, written_idx, tok):
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    with gzip.open(os.path.join(run_dir, "predictions", name.replace("/", "--") + ".jsonl.gz"), "wt") as f:
        for r, s, g, i in zip(rows, scored, written_ids, written_idx):
            f.write(json.dumps({"rec_idx": r["rec_idx"], "qid": r["qid"], "src": r["src"], "qtype": r["qtype"], "label": r["label"],
                                "logits": [round(float(x), 4) for x in s], "written": tok.decode(g)[:200], "written_idx": i}) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, few steps: checks the whole pipeline")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override config keys, e.g. --set seed=1 dtype=bfloat16")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    if cfg["method"] != "lora":
        raise SystemExit("the generative arm implements the adapter recipe only (method: lora)")
    name = cfg["name"] + ("-smoke" if args.smoke else "")
    run = Run(name, cfg | {"smoke": args.smoke}, allow_dirty=args.allow_dirty or args.smoke, sync=not args.no_sync)
    print(f"[run] {run.id}", flush=True)
    mx.random.seed(cfg["seed"])
    g = cfg.get("guard", {})
    guard = ResourceGuard(**g, log=lambda m: print(m, flush=True))
    print(f"[guard] start {guard.snapshot()} limits {g}", flush=True)

    lm, tokw, base_config = gen.load_lm(cfg["base_model"], cfg["base_revision"])
    model = gen.GenModel(lm)
    model.set_dtype(getattr(mx, cfg["dtype"]))
    apply_lora(model, cfg["lora"])                              # frozen base and output layer, trainable LoRA
    gen.freeze_output(model)
    if cfg["grad_checkpoint"]:
        gen.grad_checkpoint(model.backbone.layers[0])
    mx.eval(model.parameters())
    setup_peak = mx.get_peak_memory() / 2**30
    mx.clear_cache()
    mx.reset_peak_memory()
    print(f"[guard] after model setup: active {mx.get_active_memory()/2**30:.2f} GB, setup peak {setup_peak:.2f} GB", flush=True)
    tok = tokw._tokenizer
    enc = gen.GenEncoder(tok, cfg["max_state"], cfg["max_row"], cfg["answer_mode"])

    data_prov = {}
    recs, data_prov["train"] = load_records(cfg["train_suite"], "train")
    train_rows, skipped = build_rows(enc, recs, cfg["train_suite"])
    data_prov["train"]["skipped"] = skipped
    recs, data_prov["dev"] = load_records(cfg["dev_suite"], "development")
    dev_rows, _ = build_rows(enc, recs, cfg["dev_suite"])
    if cfg.get("train_limit"):
        pick = sorted(random.Random(cfg.get("subset_seed", 0)).sample(range(len(train_rows)), min(cfg["train_limit"], len(train_rows))))
        train_rows = [train_rows[i] for i in pick]
    if args.smoke:
        train_rows, dev_rows = train_rows[:: max(1, len(train_rows) // 240)][:240], dev_rows[:: max(1, len(dev_rows) // 120)][:120]
    n_tok = sum(len(r["ids"]) + len(r["answer_ids"][r["label"]]) for r in train_rows)
    n_ans = sum(len(r["answer_ids"][r["label"]]) for r in train_rows)
    print(f"[data] train {len(train_rows)} questions / {n_tok:,} tokens ({n_ans:,} answer tokens); dev {len(dev_rows)}; skipped {skipped}", flush=True)

    # Batches are built from prompt lengths exactly as in scripts/train.py, so in text mode a run sees the same
    # questions in the same order and takes the same number of steps as the decision run with the same seed.
    epochs = 1 if args.smoke else cfg["epochs"]
    epoch_batches = [batches(train_rows, cfg["token_budget"], shuffle=True, seed=cfg["seed"] * 1000 + e) for e in range(epochs)]
    total_micro = sum(len(b) for b in epoch_batches)
    total_steps = math.ceil(total_micro / cfg["accum_microbatches"])
    warm = max(1, int(cfg["warmup_frac"] * total_steps))
    peak = cfg["backbone_lr"]
    opt = optim.AdamW(learning_rate=optim.join_schedules([optim.linear_schedule(0.0, peak, warm),
                                                          optim.cosine_decay(peak, max(1, total_steps - warm), peak * cfg["min_lr_frac"])], [warm]),
                      weight_decay=cfg["weight_decay"])
    trainable = count(model.trainable_parameters())
    run.write("data.json", data_prov | {"train_questions": len(train_rows), "train_tokens": n_tok, "answer_tokens": n_ans,
                                        "dev_questions": len(dev_rows), "answer_mode": cfg["answer_mode"]})
    print(f"[model] trainable {trainable/1e6:.1f}M of {count(model.parameters())/1e6:.1f}M params; {total_steps} steps", flush=True)

    loss_and_grad = nn.value_and_grad(model, gen.gen_loss)
    eval_every = max(1, total_steps // (cfg["evals_per_epoch"] * epochs)) if not args.smoke else max(1, total_steps // 2)
    ckpt_root = os.path.join(REPO_ROOT, "checkpoints", run.id)
    best = {"accuracy": -1}
    step, t0, win = 0, time.time(), {"loss": 0.0, "correct": 0, "rows": 0, "tokens": 0, "t": time.time()}
    meta = {"name": cfg["name"], "kind": "generative", "answer_mode": cfg["answer_mode"], "method": cfg["method"],
            "base_model": cfg["base_model"], "base_revision": cfg["base_revision"], "max_state": cfg["max_state"],
            "max_row": cfg["max_row"], "run_id": run.id, "weights": "adapter", "lora": cfg["lora"], "base_dtype": cfg["dtype"]}

    def evaluate_written(rows):
        model.eval()
        with gen.no_checkpoint():
            out = gen.read_rows(model, lm, tokw, rows, write=True, score=False, guard=guard)
        model.train()
        idx = [gen.parse_written(r, t, tok, enc.eos_id) for r, t in zip(rows, out["written"])]
        return gen.written_summary(rows, idx) | {"sec_per_question": out["write_sec"]}

    grads_acc, micro_in_step, rows_in_step = None, 0, 0
    model.train()
    try:
        for e, eb in enumerate(epoch_batches):
            for idx in eb:
                guard.check()
                batch = [train_rows[i] for i in idx]
                ids, pos, tgt, own = gen.collate_gen(batch, enc.pad_id)
                (loss, correct), grads = loss_and_grad(model, ids, pos, tgt, own, len(batch))
                mx.eval(loss, correct, grads)                   # together, as in train.py, or checkpointing is defeated
                grads_acc = grads if grads_acc is None else tree_map(mx.add, grads_acc, grads)
                micro_in_step += 1; rows_in_step += len(batch)
                win["loss"] += loss.item(); win["correct"] += correct.item(); win["rows"] += len(batch)
                win["tokens"] += sum(len(r["ids"]) + len(r["answer_ids"][r["label"]]) for r in batch)
                if micro_in_step < cfg["accum_microbatches"]:
                    continue
                grads_step = tree_map(lambda x: x / rows_in_step, grads_acc)
                grads_step, gnorm = optim.clip_grad_norm(grads_step, cfg["grad_clip"])
                opt.update(model, grads_step)
                mx.eval(model.parameters(), opt.state)
                grads_acc, micro_in_step, rows_in_step = None, 0, 0
                step += 1

                if step % cfg["log_every"] == 0 or step == total_steps:
                    dt = time.time() - win["t"]
                    rec = {"step": step, "epoch": e, "loss": win["loss"] / win["rows"], "accuracy": win["correct"] / win["rows"],
                           "grad_norm": gnorm.item(), "lr_backbone": float(opt.learning_rate.item()),
                           "tokens_per_sec": win["tokens"] / dt, "questions_per_sec": win["rows"] / dt,
                           "peak_mem_gb": mx.get_peak_memory() / 2**30, "elapsed_min": (time.time() - t0) / 60}
                    run.log("train_log.jsonl", rec, step)
                    eta = (time.time() - t0) / step * (total_steps - step) / 60
                    print(f"[train] step {step}/{total_steps} loss {rec['loss']:.4f} acc {rec['accuracy']:.3f} "
                          f"{rec['tokens_per_sec']:.0f} tok/s mem {rec['peak_mem_gb']:.1f}GB eta {eta:.0f}m", flush=True)
                    win = {"loss": 0.0, "correct": 0, "rows": 0, "tokens": 0, "t": time.time()}

                if step % eval_every == 0 or step == total_steps:
                    m = evaluate_written(dev_rows)
                    m.update(step=step, epoch=e)
                    is_best = m["accuracy"] > best["accuracy"]
                    m["best"] = is_best
                    run.log("evals.jsonl", {k: v for k, v in m.items() if k != "by_type"}, step)
                    print(f"[eval] step {step} dev written acc {m['accuracy']:.4f} invalid {m['invalid']:.4f} "
                          f"{m['sec_per_question']*1000:.0f} ms/q{' (best)' if is_best else ''}", flush=True)
                    if is_best:
                        best = m
                        save_checkpoint(model, os.path.join(ckpt_root, "best"), tok, base_config, meta | {"step": step})
                    run.sync(f"step {step}, dev written acc {m['accuracy']:.4f}")
                if step % cfg["log_every"] == 0:
                    run.log("resources.jsonl", {"step": step} | guard.snapshot(), step)
    except ResourceAbort as err:
        snap = guard.snapshot()
        print(f"[guard] ABORT: {err} {snap}", flush=True)
        run.write("aborted.json", {"reason": str(err), "step": step, "resources": snap,
                                   "best_dev_accuracy": best.get("accuracy"), "best_checkpoint_kept": best.get("accuracy", -1) >= 0})
        run.sync(f"aborted by resource guard at step {step}: {err}")
        sys.exit(3)

    # Final: reload the best checkpoint; fit a temperature to the scored distribution on the calibration split; read
    # every final split once, both written and scored. No more training, so cached passes run without checkpointing.
    try:
        with gen.no_checkpoint():
            del model, lm
            mx.clear_cache()
            model, lm, tokw, _ = gen.load_gen_checkpoint(os.path.join(ckpt_root, "best"))
            model.eval()
            suite, split = cfg["calibration"]
            recs, cal_prov = load_records(suite, split)
            cal_rows, _ = build_rows(enc, recs, suite)
            if args.smoke:
                cal_rows = cal_rows[:100]
            cal = gen.read_rows(model, lm, tokw, cal_rows, write=False, score=True, guard=guard, chunk=cfg["score_chunk"])
            T = fit_temperature([{"logits": s, "label": r["label"]} for r, s in zip(cal_rows, cal["scores"])])
            final = {"run_id": run.id, "kind": "generative", "answer_mode": cfg["answer_mode"], "best_step": best["step"],
                     "temperature": T, "calibration_data": cal_prov, "train_minutes": (time.time() - t0) / 60,
                     "trainable_params": trainable, "evals": {}}
            for suite, split in cfg["final_evals"]:
                recs, prov = load_records(suite, split)
                rows, skipped = build_rows(enc, recs, suite)
                if args.smoke:
                    rows = rows[:100]
                out = gen.read_rows(model, lm, tokw, rows, write=True, score=True, guard=guard, chunk=cfg["score_chunk"])
                written_ids, scores, w_sec, s_sec = out["written"], out["scores"], out["write_sec"], out["score_sec"]
                written_idx = [gen.parse_written(r, t, tok, enc.eos_id) for r, t in zip(rows, written_ids)]
                preds = [{"logits": s, "label": r["label"], "src": r["src"], "qtype": r["qtype"]} for r, s in zip(rows, scores)]
                write_predictions(run.dir, f"{suite}/{split}", rows, scores, written_ids, written_idx, tok)
                final["evals"][f"{suite}/{split}"] = {"written": gen.written_summary(rows, written_idx), "raw": summarize(preds),
                                                       "calibrated": summarize(preds, T), "sec_per_question": w_sec,
                                                       "score_sec_per_question": s_sec, "data": prov | {"skipped": skipped}}
                r = final["evals"][f"{suite}/{split}"]
                print(f"[final] {suite}/{split}: written acc {r['written']['accuracy']:.4f} invalid {r['written']['invalid']:.4f} | "
                      f"scored acc {r['raw']['accuracy']:.4f} ece raw {r['raw']['ece']:.4f} / T={T:.2f} {r['calibrated']['ece']:.4f} | "
                      f"{w_sec*1000:.0f} ms/q written", flush=True)
    except ResourceAbort as err:
        print(f"[guard] ABORT during final evaluation: {err}", flush=True)
        run.write("aborted.json", {"reason": str(err), "step": step, "phase": "final evaluation", "resources": guard.snapshot(),
                                   "best_dev_accuracy": best.get("accuracy"), "best_checkpoint_kept": True})
        run.sync(f"aborted by resource guard during final evaluation: {err}")
        sys.exit(3)
    run.write("final.json", final)
    publish_dir = os.path.join(ckpt_root, "publish")
    save_checkpoint(model, publish_dir, tok, base_config, meta | {"step": best["step"], "temperature": T})
    run.write("checkpoint.json", {"local_dir": os.path.relpath(publish_dir, REPO_ROOT), "published": None})
    run.sync("final evaluation")
    print(f"[done] {run.id} -> {publish_dir}", flush=True)


if __name__ == "__main__":
    main()
