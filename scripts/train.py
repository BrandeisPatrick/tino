"""Train a tino decision model. Method 0 (full fine-tune) for now; adapter and distillation reuse this loop.

    uv run python scripts/train.py --config configs/tino-finetuned.json
    uv run python scripts/train.py --config configs/tino-finetuned.json --smoke   # tiny end-to-end check
"""
import argparse
import gzip
import random
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from tino.data import batches, build_rows, load_records
from tino.distil import attach, collate_teacher, distil_loss, load_teacher_labels
from tino.evaluate import predict
from tino.format import Encoder
from tino.guard import ResourceAbort, ResourceGuard
from tino.metrics import fit_temperature, summarize
from tino.model import apply_lora, collate, load_base, save_checkpoint
from tino.track import REPO_ROOT, Run


def grad_checkpoint(layer):
    fn = type(layer).__call__

    def checkpointed(model, *args, **kwargs):
        def inner(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)
        return mx.checkpoint(inner)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed


def loss_fn(model, ids, dec, opt, mask, labels):
    z = model(ids, dec, opt, mask)
    ce = mx.logsumexp(z, axis=-1) - mx.take_along_axis(z, labels[:, None], axis=-1).squeeze(-1)
    return ce.sum(), (mx.argmax(z, axis=-1) == labels).sum()


def make_ce_loss(brier_weight=0.0):
    """Cross-entropy, optionally plus brier_weight x multi-class Brier score (a proper scoring rule that punishes
    confident mistakes less harshly than log loss)."""
    if not brier_weight:
        return loss_fn

    def fn(model, ids, dec, opt, mask, labels):
        z = model(ids, dec, opt, mask)
        logp = z - mx.logsumexp(z, axis=-1, keepdims=True)
        ce = -mx.take_along_axis(logp, labels[:, None], axis=-1).squeeze(-1)
        onehot = (mx.arange(z.shape[-1])[None, :] == labels[:, None]).astype(mx.float32)
        brier = ((mx.exp(logp) - onehot) ** 2).sum(axis=-1)
        return (ce + brier_weight * brier).sum(), (mx.argmax(z, axis=-1) == labels).sum()
    return fn


def write_predictions(run_dir, name, rows, preds):
    """Per-question predictions (gzipped JSONL) so runs can be compared question by question."""
    from tino.metrics import softmax
    os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)
    with gzip.open(os.path.join(run_dir, "predictions", name.replace("/", "--") + ".jsonl.gz"), "wt") as f:
        for r, p in zip(rows, preds):
            f.write(json.dumps({"rec_idx": r["rec_idx"], "qid": r["qid"], "src": r["src"], "qtype": r["qtype"], "label": r["label"],
                                "logits": [round(float(x), 4) for x in p["logits"]]}) + "\n")


def make_distil_loss(alpha):
    def fn(model, ids, dec, opt, mask, labels, teacher_p, has_teacher):
        z = model(ids, dec, opt, mask)
        return distil_loss(z, labels, teacher_p, has_teacher, alpha).sum(), (mx.argmax(z, axis=-1) == labels).sum()
    return fn


def count(params):
    return sum(v.size for _, v in tree_flatten(params))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true", help="tiny subset, few steps: checks the whole pipeline")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--no-sync", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override config keys, e.g. --set epochs=1 dtype=bfloat16")
    args = ap.parse_args()

    cfg = json.load(open(args.config))
    for kv in args.set:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except json.JSONDecodeError:
            cfg[k] = v
    name = cfg["name"] + ("-smoke" if args.smoke else "")
    run = Run(name, cfg | {"smoke": args.smoke}, allow_dirty=args.allow_dirty or args.smoke, sync=not args.no_sync)
    print(f"[run] {run.id}", flush=True)
    mx.random.seed(cfg["seed"])
    g = cfg.get("guard", {})
    guard = ResourceGuard(**g, log=lambda m: print(m, flush=True))
    print(f"[guard] start {guard.snapshot()} limits {g}", flush=True)

    model, tok, base_config = load_base(cfg["base_model"], cfg["base_revision"], cfg["pointer_dim"])
    model.set_dtype(getattr(mx, cfg["dtype"]))
    model.head.set_dtype(mx.float32)
    if cfg["method"] == "lora":
        apply_lora(model, cfg["lora"])                          # frozen base, trainable LoRA + head
    elif cfg["freeze_embeddings"]:
        model.backbone.embed_tokens.freeze()
        model.backbone.embed_tokens.set_dtype(mx.bfloat16)   # frozen: no need for fp32, saves ~0.3 GB
    if cfg["grad_checkpoint"]:
        grad_checkpoint(model.backbone.layers[0])
    # Materialize dtype conversions now: left lazy, the bf16->fp32 copy runs inside the first training step with both
    # copies alive (a one-off 9 GB spike in the first adapter smoke run). Reset the peak so logs show training memory.
    mx.eval(model.parameters())
    setup_peak = mx.get_peak_memory() / 2**30
    mx.clear_cache()
    mx.reset_peak_memory()
    print(f"[guard] after model setup: active {mx.get_active_memory()/2**30:.2f} GB, setup peak {setup_peak:.2f} GB", flush=True)
    enc = Encoder(tok, cfg["max_state"], cfg["max_row"])

    data_prov = {}
    recs, data_prov["train"] = load_records(cfg["train_suite"], "train")
    train_rows, skipped = build_rows(enc, recs, cfg["train_suite"])
    data_prov["train"]["skipped"] = skipped
    recs, data_prov["dev"] = load_records(cfg["dev_suite"], "development")
    dev_rows, _ = build_rows(enc, recs, cfg["dev_suite"])
    dcfg = cfg.get("distil")
    if dcfg:
        labels, man = load_teacher_labels(dcfg["labels_repo"], dcfg["labels_prefix"], dcfg["labels_revision"], cfg["train_suite"], "train")
        t_temp = man["temperature"] if dcfg.get("teacher_temperature", "manifest") == "manifest" else float(dcfg["teacher_temperature"])
        cov = attach(train_rows, labels, t_temp)
        data_prov["teacher"] = {"repo": dcfg["labels_repo"], "prefix": dcfg["labels_prefix"], "revision": dcfg["labels_revision"],
                                "teacher": man["teacher"], "teacher_revision": man["teacher_revision"], "prompt_sha": man["prompt_sha"],
                                "temperature": t_temp, "alpha": dcfg["alpha"], "coverage": cov}
        print(f"[distil] teacher {man['teacher']} T={t_temp:.3f} alpha={dcfg['alpha']} coverage {cov}", flush=True)
    if cfg.get("train_limit"):
        # Fixed subset (subset_seed, not the training seed), so seeds and methods at the same size see the same questions.
        pick = sorted(random.Random(cfg.get("subset_seed", 0)).sample(range(len(train_rows)), min(cfg["train_limit"], len(train_rows))))
        train_rows = [train_rows[i] for i in pick]
    if args.smoke:
        train_rows, dev_rows = train_rows[:: max(1, len(train_rows) // 240)][:240], dev_rows[:: max(1, len(dev_rows) // 120)][:120]
    n_tok = sum(len(r["ids"]) for r in train_rows)
    print(f"[data] train {len(train_rows)} questions / {n_tok:,} tokens; dev {len(dev_rows)}; skipped {skipped}", flush=True)

    epochs = 1 if args.smoke else cfg["epochs"]
    epoch_batches = [batches(train_rows, cfg["token_budget"], shuffle=True, seed=cfg["seed"] * 1000 + e) for e in range(epochs)]
    total_micro = sum(len(b) for b in epoch_batches)
    total_steps = math.ceil(total_micro / cfg["accum_microbatches"])
    warm = max(1, int(cfg["warmup_frac"] * total_steps))

    def sched(peak):
        return optim.join_schedules([optim.linear_schedule(0.0, peak, warm),
                                     optim.cosine_decay(peak, max(1, total_steps - warm), peak * cfg["min_lr_frac"])], [warm])

    if cfg.get("backbone_optimizer", "adamw") == "adafactor":
        # Factored second moments, no first moment: optimizer state is a few MB instead of 2 x fp32 weights (~3.5 GB).
        backbone_opt = optim.Adafactor(learning_rate=sched(cfg["backbone_lr"]), relative_step=False,
                                       scale_parameter=False, beta_1=None, clip_threshold=1.0)
    else:
        backbone_opt = optim.AdamW(learning_rate=sched(cfg["backbone_lr"]), weight_decay=cfg["weight_decay"])
    opt = optim.MultiOptimizer(
        [optim.AdamW(learning_rate=sched(cfg["head_lr"]), weight_decay=cfg.get("head_weight_decay", 0.0)), backbone_opt],
        [lambda path, _: path.startswith("head")])
    trainable = count(model.trainable_parameters())
    run.write("data.json", data_prov | {"train_questions": len(train_rows), "train_tokens": n_tok, "dev_questions": len(dev_rows)})
    print(f"[model] trainable {trainable/1e6:.1f}M of {count(model.parameters())/1e6:.1f}M params; {total_steps} steps", flush=True)

    loss_and_grad = nn.value_and_grad(model, make_distil_loss(dcfg["alpha"]) if dcfg else make_ce_loss(cfg.get("brier_weight", 0.0)))
    eval_every = max(1, total_steps // (cfg["evals_per_epoch"] * epochs)) if not args.smoke else max(1, total_steps // 2)
    ckpt_root = os.path.join(REPO_ROOT, "checkpoints", run.id)
    best = {"accuracy": -1}
    step, t0, win = 0, time.time(), {"loss": 0.0, "correct": 0, "rows": 0, "tokens": 0, "t": time.time()}
    meta = {"name": cfg["name"], "method": cfg["method"], "base_model": cfg["base_model"], "base_revision": cfg["base_revision"],
            "pointer_dim": cfg["pointer_dim"], "max_state": cfg["max_state"], "max_row": cfg["max_row"], "run_id": run.id,
            "temperature": 1.0, "weights": "adapter" if cfg["method"] == "lora" else "full", "lora": cfg.get("lora"),
            "base_dtype": cfg["dtype"]}

    def evaluate_dev():
        model.eval()
        preds, sec = predict(model, dev_rows, enc.pad_id, cfg['eval_token_budget'], guard)
        model.train()
        return summarize(preds) | {"sec_per_question": sec}

    grads_acc, micro_in_step, rows_in_step = None, 0, 0
    model.train()
    try:
        for e, eb in enumerate(epoch_batches):
            for idx in eb:
                guard.check()
                batch = [train_rows[i] for i in idx]
                ids, dec, o, mask, lab = collate(batch, enc.pad_id)
                extra = collate_teacher(batch, o.shape[1]) if dcfg else ()
                (loss, correct), grads = loss_and_grad(model, ids, dec, o, mask, lab, *extra)
                # Evaluate loss and grads together. Reading loss.item() first runs the forward pass alone and keeps every
                # activation the pending backward needs alive, defeating gradient checkpointing (9.2 GB vs 3.9 GB peak).
                mx.eval(loss, correct, grads)
                grads_acc = grads if grads_acc is None else tree_map(mx.add, grads_acc, grads)
                micro_in_step += 1; rows_in_step += len(batch)
                win["loss"] += loss.item(); win["correct"] += correct.item(); win["rows"] += len(batch)
                win["tokens"] += sum(len(r["ids"]) for r in batch)
                if micro_in_step < cfg["accum_microbatches"]:
                    continue
                g = tree_map(lambda x: x / rows_in_step, grads_acc)
                g, gnorm = optim.clip_grad_norm(g, cfg["grad_clip"])
                opt.update(model, g)
                mx.eval(model.parameters(), opt.state)
                grads_acc, micro_in_step, rows_in_step = None, 0, 0
                step += 1

                if step % cfg["log_every"] == 0 or step == total_steps:
                    dt = time.time() - win["t"]
                    rec = {"step": step, "epoch": e, "loss": win["loss"] / win["rows"], "accuracy": win["correct"] / win["rows"],
                           "grad_norm": gnorm.item(), "lr_backbone": float(opt.optimizers[1].learning_rate.item()),
                           "tokens_per_sec": win["tokens"] / dt, "questions_per_sec": win["rows"] / dt,
                           "peak_mem_gb": mx.get_peak_memory() / 2**30, "elapsed_min": (time.time() - t0) / 60}
                    run.log("train_log.jsonl", rec, step)
                    eta = (time.time() - t0) / step * (total_steps - step) / 60
                    print(f"[train] step {step}/{total_steps} loss {rec['loss']:.4f} acc {rec['accuracy']:.3f} "
                          f"{rec['tokens_per_sec']:.0f} tok/s mem {rec['peak_mem_gb']:.1f}GB eta {eta:.0f}m", flush=True)
                    win = {"loss": 0.0, "correct": 0, "rows": 0, "tokens": 0, "t": time.time()}

                if step % eval_every == 0 or step == total_steps:
                    m = evaluate_dev()
                    m.update(step=step, epoch=e)
                    is_best = m["accuracy"] > best["accuracy"]
                    m["best"] = is_best
                    run.log("evals.jsonl", {k: v for k, v in m.items() if k not in ("by_source", "by_type")} |
                            {"by_source": m["by_source"]}, step)
                    print(f"[eval] step {step} dev acc {m['accuracy']:.4f} nll {m['nll']:.4f} ece {m['ece']:.4f}"
                          f"{' (best)' if is_best else ''}", flush=True)
                    if is_best:
                        best = m
                        save_checkpoint(model, os.path.join(ckpt_root, "best"), tok, base_config, meta | {"step": step})
                    run.sync(f"step {step}, dev acc {m['accuracy']:.4f}")
                if step % cfg["log_every"] == 0:
                    run.log("resources.jsonl", {"step": step} | guard.snapshot(), step)
    except ResourceAbort as err:
        snap = guard.snapshot()
        print(f"[guard] ABORT: {err} {snap}", flush=True)
        run.write("aborted.json", {"reason": str(err), "step": step, "resources": snap,
                                   "best_dev_accuracy": best.get("accuracy"), "best_checkpoint_kept": best.get("accuracy", -1) >= 0})
        run.sync(f"aborted by resource guard at step {step}: {err}")
        sys.exit(3)

    # Final: reload best, fit temperature on the calibration split, read each final eval set once.
    from tino.model import load_checkpoint
    model, _, _ = load_checkpoint(os.path.join(ckpt_root, "best"))
    if cfg.get("publish_dtype"):
        model.backbone.set_dtype(getattr(mx, cfg["publish_dtype"]))   # evaluate exactly what gets published
    model.eval()
    suite, split = cfg["calibration"]
    recs, cal_prov = load_records(suite, split)
    cal_rows, _ = build_rows(enc, recs, suite)
    if args.smoke:
        cal_rows = cal_rows[:100]
    cal_preds, _ = predict(model, cal_rows, enc.pad_id, cfg['eval_token_budget'], guard)
    T = fit_temperature(cal_preds)
    final = {"run_id": run.id, "best_step": best["step"], "temperature": T, "calibration_data": cal_prov,
             "train_minutes": (time.time() - t0) / 60, "trainable_params": trainable, "evals": {}}
    for suite, split in cfg["final_evals"]:
        recs, prov = load_records(suite, split)
        rows, skipped = build_rows(enc, recs, suite)
        if args.smoke:
            rows = rows[:100]
        preds, sec = predict(model, rows, enc.pad_id, cfg['eval_token_budget'], guard)
        write_predictions(run.dir, f"{suite}/{split}", rows, preds)
        final["evals"][f"{suite}/{split}"] = {"raw": summarize(preds), "calibrated": summarize(preds, T),
                                               "sec_per_question": sec, "data": prov | {"skipped": skipped}}
        r = final["evals"][f"{suite}/{split}"]
        print(f"[final] {suite}/{split}: acc {r['raw']['accuracy']:.4f} brier {r['raw']['brier']:.4f} "
              f"ece raw {r['raw']['ece']:.4f} / T={T:.2f} {r['calibrated']['ece']:.4f}", flush=True)
    run.write("final.json", final)
    publish_dir = os.path.join(ckpt_root, "publish")
    save_checkpoint(model, publish_dir, tok, base_config, meta | {"step": best["step"], "temperature": T},
                    dtype=getattr(mx, cfg["publish_dtype"]) if cfg.get("publish_dtype") else None)
    run.write("checkpoint.json", {"local_dir": os.path.relpath(publish_dir, REPO_ROOT), "published": None})
    run.sync("final evaluation")
    print(f"[done] {run.id} -> {publish_dir}", flush=True)


if __name__ == "__main__":
    main()
