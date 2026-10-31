"""The generative counterpart of the decision model: same backbone, same input row, same adapter recipe, but it
answers by writing the chosen option instead of pointing at it (RESEARCH.md, "Later: are decision models more
parameter-efficient?").

The input is the decision model's row, unchanged. The answer follows <DECIDE> as ordinary tokens and ends with the
tokenizer's end-of-text token:

    <STATE> ... <Q> ... (<OPT> option </OPT>)* <DECIDE> answer <eos>

answer_mode "text": the answer is the option's text exactly as shown inside its <OPT> span (same tokens).
answer_mode "letter": each option is shown as "A. text" and the answer is its letter: a control that keeps the
language-model head but removes nearly all of the writing.

Training loss is the cross-entropy of the answer tokens only. Evaluation reads the model two ways:
- written: greedy generation, parsed back to an option. An answer that matches no option counts as wrong.
  Questions are read one at a time with their own KV cache: the prompt runs once for both readings.
- scored: the log-probability of each option's answer given the prompt, a distribution over the options that gives
  accuracy, log-loss and calibration on the same footing as the decision model.
"""
import json
import os
import re
import string
import time
from collections import defaultdict
from contextlib import contextmanager

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .format import Encoder, question_options, render
from .model import apply_lora

_LETTERS = string.ascii_uppercase


def letter(i):
    """A..Z, then AA, AB, ... for questions with more than 26 options."""
    return _LETTERS[i] if i < 26 else _LETTERS[i // 26 - 1] + _LETTERS[i % 26]


class GenEncoder(Encoder):
    """The decision model's rows plus, for every option, the answer it would be written as."""

    def __init__(self, tokenizer, max_state=384, max_row=1536, mode="text"):
        super().__init__(tokenizer, max_state, max_row)
        if mode not in ("text", "letter"):
            raise ValueError(f"answer_mode must be 'text' or 'letter', not {mode!r}")
        self.mode = mode
        self.eos_id = tokenizer.eos_token_id

    def question_row(self, state_ids, q):
        keys, opts = question_options(q)
        shown = [f"{letter(i)}. {o}" for i, o in enumerate(opts)] if self.mode == "letter" else opts
        answers = [letter(i) for i in range(len(opts))] if self.mode == "letter" else opts
        ids = list(state_ids) + [self.sid["q"]] + self.user_tokens(render(q.get("instructions")))
        opt_idx = []
        for o in shown:
            ids += [self.sid["opt"]] + self.user_tokens(o) + [self.sid["end_opt"]]
            opt_idx.append(len(ids) - 1)
        ids.append(self.sid["decide"])
        return {"ids": ids, "opt_idx": opt_idx, "decide_idx": len(ids) - 1, "keys": keys, "answers": answers,
                "answer_ids": [self.user_tokens(a) + [self.eos_id] for a in answers]}


class GenModel(nn.Module):
    """The language model itself: backbone plus its output layer (tied to the embeddings in Qwen3 0.6B and 1.7B)."""

    def __init__(self, lm):
        super().__init__()
        self.backbone = lm.model
        if hasattr(lm, "lm_head"):
            self.lm_head = lm.lm_head

    def project(self, h):
        head = getattr(self, "lm_head", None)
        return (head(h) if head is not None else self.backbone.embed_tokens.as_linear(h)).astype(mx.float32)


def freeze_output(model):
    """apply_lora freezes the backbone; an untied output layer must be frozen too (Qwen3 0.6B/1.7B have none)."""
    if getattr(model, "lm_head", None) is not None:
        model.lm_head.freeze()


_CKPT = {"on": True}


def grad_checkpoint(layer):
    """Gradient checkpointing for every block of layer's class, switchable off for cached generation and scoring."""
    fn = type(layer).__call__

    def checkpointed(model, *args, **kwargs):
        if not _CKPT["on"]:
            return fn(model, *args, **kwargs)

        def inner(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)
        return mx.checkpoint(inner)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed


@contextmanager
def no_checkpoint():
    prev = _CKPT["on"]
    _CKPT["on"] = False
    try:
        yield
    finally:
        _CKPT["on"] = prev


def collate_gen(batch, pad_id):
    """Prompt + gold answer per row, right-padded. -> ids [B, L], flat positions whose next token is an answer token,
    those target tokens, and the row each target belongs to."""
    seqs = [r["ids"] + r["answer_ids"][r["label"]] for r in batch]
    L = max(len(s) for s in seqs)
    ids = np.full((len(batch), L), pad_id, dtype=np.int32)
    pos, tgt, own = [], [], []
    for b, (r, s) in enumerate(zip(batch, seqs)):
        ids[b, : len(s)] = s
        for t in range(len(r["ids"]), len(s)):
            pos.append(b * L + t - 1)
            tgt.append(s[t])
            own.append(b)
    return (mx.array(ids), mx.array(np.array(pos, dtype=np.int32)), mx.array(np.array(tgt, dtype=np.int32)),
            mx.array(np.array(own, dtype=np.int32)))


def gen_loss(model, ids, pos, tgt, own, n_rows):
    """Summed answer-token cross-entropy, and how many rows get every answer token right under teacher forcing.
    Logits are computed only at answer positions: the full [B, L, vocab] tensor would be over 1 GB in fp32."""
    h = model.backbone(ids)
    B, L, D = h.shape
    z = model.project(h.reshape(B * L, D)[pos])
    ce = mx.logsumexp(z, axis=-1) - mx.take_along_axis(z, tgt[:, None], axis=-1).squeeze(-1)
    miss = mx.zeros((n_rows,)).at[own].add((mx.argmax(z, axis=-1) != tgt).astype(mx.float32))
    return ce.sum(), (miss == 0).sum()


def _norm(s):
    return re.sub(r"\s+", " ", s).strip().strip(".").strip().lower()


def parse_written(row, gen_ids, tok, eos_id=None):
    """-> option index the generated answer names, or -1. Exact tokens first, then the text ignoring case, spacing
    and a trailing period, then (text answers) the option's name before ': ' when it names exactly one option."""
    gen_ids = [t for t in gen_ids if t != eos_id]
    for k, a in enumerate(row["answer_ids"]):
        if gen_ids == a[:-1]:
            return k
    text = _norm(tok.decode(gen_ids))
    hits = [k for k, a in enumerate(row["answers"]) if _norm(a) == text]
    if not hits:
        hits = [k for k, a in enumerate(row["answers"]) if _norm(a.split(": ", 1)[0]) == text]
    return hits[0] if len(hits) == 1 else -1


def written_summary(rows, idx):
    if not rows:
        return {"n": 0}
    ok = [int(i == r["label"]) for r, i in zip(rows, idx)]
    by_src, by_type = defaultdict(list), defaultdict(list)
    for r, o in zip(rows, ok):
        by_src[r.get("src") or "?"].append(o)
        by_type[r["qtype"]].append(o)
    return {"n": len(rows), "accuracy": float(np.mean(ok)), "invalid": float(np.mean([i == -1 for i in idx])),
            "macro_source_accuracy": float(np.mean([np.mean(v) for v in by_src.values()])),
            "by_source": {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(by_src.items())},
            "by_type": {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(by_type.items())}}


def _prefill(model, lm, ids):
    """Run the prompt once. -> (cache, log-probs of the first answer token [1, V], prompt keys/values, prompt length)."""
    from mlx_lm.models.cache import make_prompt_cache
    cache = make_prompt_cache(lm)
    h = model.backbone(mx.array(ids)[None], cache=cache)
    first = model.project(h[0, -1:])
    first = first - mx.logsumexp(first, axis=-1, keepdims=True)
    P = cache[0].offset
    base = [(c.keys[..., :P, :], c.values[..., :P, :]) for c in cache]
    mx.eval(first, base)
    return cache, first, base, P


def _restore(cache, base, P, n=1):
    """Reset the cache to the prompt, copied n times along the batch axis."""
    for c, (k, v) in zip(cache, base):
        c.keys, c.values, c.offset = (k, v, P) if n == 1 else (mx.repeat(k, n, axis=0), mx.repeat(v, n, axis=0), P)


def _greedy(model, cache, first, stop, max_new):
    toks, t = [], int(mx.argmax(first[0]).item())
    while t not in stop and len(toks) < max_new:
        toks.append(t)
        if len(toks) == max_new:
            break
        h = model.backbone(mx.array([[t]]), cache=cache)
        t = int(mx.argmax(model.project(h[0, -1])).item())
    return toks


def _score(model, cache, first, base, P, answers, chunk):
    scores = []
    for s in range(0, len(answers), chunk):
        part = answers[s:s + chunk]
        lp = first[0, mx.array([a[0] for a in part])]
        M = max(len(a) for a in part)
        if M > 1:
            _restore(cache, base, P, len(part))
            opt = np.full((len(part), M - 1), part[0][-1], dtype=np.int32)
            for i, a in enumerate(part):
                opt[i, : len(a) - 1] = a[:-1]
            h = model.backbone(mx.array(opt), cache=cache)
            pos = [i * (M - 1) + j - 1 for i, a in enumerate(part) for j in range(1, len(a))]
            tgt = [a[j] for a in part for j in range(1, len(a))]
            own = [i for i, a in enumerate(part) for j in range(1, len(a))]
            z = model.project(h.reshape(len(part) * (M - 1), -1)[mx.array(pos)])
            tok_lp = mx.take_along_axis(z, mx.array(tgt)[:, None], axis=-1).squeeze(-1) - mx.logsumexp(z, axis=-1)
            lp = lp + mx.zeros((len(part),)).at[mx.array(own)].add(tok_lp)
        mx.eval(lp)
        scores += np.array(lp).tolist()
    return scores


def read_rows(model, lm, tokw, rows, write=True, score=True, guard=None, chunk=4):
    """One question at a time, each with its own KV cache (at most about 0.35 GB for a 1,536-token prompt at 0.6B in
    fp32; batched generation of 16 long prompts ran out of GPU memory in the first smoke run). The prompt runs once;
    the written answer is greedy decoding from it, and each option's score is the log-probability of its full answer
    (end token included), with the prompt cache copied for up to `chunk` options at a time.
    -> {"written": token ids per row (end token dropped) or None, "scores": per-option log-probs per row or None,
        "write_sec": seconds per question to read the prompt and write the answer, "score_sec": seconds per question
        to score the options}."""
    stop = set(tokw.eos_token_ids)
    written, scores, t_write, t_score = [], [], 0.0, 0.0
    for n, r in enumerate(rows):
        if guard is not None and n % 32 == 0:
            guard.check()
        t0 = time.time()
        cache, first, base, P = _prefill(model, lm, r["ids"])
        t_pre = time.time() - t0
        if score:
            t0 = time.time()
            scores.append(_score(model, cache, first, base, P, r["answer_ids"], chunk))
            t_score += time.time() - t0 + (0.0 if write else t_pre)
        if write:
            t0 = time.time()
            if score:
                _restore(cache, base, P)
            written.append(_greedy(model, cache, first, stop, max(len(a) for a in r["answer_ids"]) + 1))
            t_write += time.time() - t0 + t_pre
        del cache, base
        if n % 64 == 63:
            mx.clear_cache()
    mx.clear_cache()
    k = max(len(rows), 1)
    return {"written": written if write else None, "scores": scores if score else None,
            "write_sec": t_write / k, "score_sec": t_score / k}


def load_lm(repo, revision=None):
    """-> (full mlx_lm model, mlx_lm tokenizer wrapper, base config)."""
    from mlx_lm import load
    return load(repo, revision=revision, return_config=True)


def load_gen_checkpoint(path):
    """Rebuild a generative tino checkpoint saved by scripts/train_gen.py (adapter weights only)."""
    meta = json.load(open(os.path.join(path, "tino_config.json")))
    lm, tokw, _ = load_lm(meta["base_model"], meta["base_revision"])
    model = GenModel(lm)
    model.set_dtype(getattr(mx, meta.get("base_dtype", "float32")))
    apply_lora(model, meta["lora"])
    freeze_output(model)
    model.load_weights(os.path.join(path, "model.safetensors"), strict=False)
    mx.eval(model.parameters())
    return model, lm, tokw, meta
