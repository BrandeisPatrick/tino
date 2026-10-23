"""An autoregressive LLM as a decision teacher.

Each question becomes an ordinary multiple-choice chat prompt (state, question, lettered options, "reply with the
letter"). One forward pass; the next-token logits at the start of the assistant's reply, restricted to the option
letters and renormalized, are the teacher's distribution over the options. No text is generated.

Questions with more than 26 options (banking77's 77 intents) cannot be lettered with single tokens, so they get no
teacher distribution; the distillation loss falls back to the gold label for them.
"""
import string

import mlx.core as mx
import numpy as np

from .format import question_options, render

LETTERS = list(string.ascii_uppercase)


def load_teacher(repo, revision=None):
    from mlx_lm import load
    model, tok = load(repo, revision=revision)
    tm = model.language_model if hasattr(model, "language_model") else model
    backbone = tm.model
    head = tm.lm_head if hasattr(tm, "lm_head") else backbone.embed_tokens.as_linear
    letter_ids = []
    for L in LETTERS:
        ids = tok.encode(L, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"teacher tokenizer splits {L!r} into {ids}")
        letter_ids.append(ids[0])
    return {"model": model, "backbone": backbone, "head": head, "tok": tok, "letter_ids": mx.array(letter_ids)}


def prompt_ids(teacher, rec, q):
    """-> token ids of the chat prompt, or None when the question has more options than letters."""
    keys, opts = question_options(q)
    if len(opts) > len(LETTERS):
        return None
    lines = [f"{LETTERS[i]}. {o}" for i, o in enumerate(opts)]
    order = " Options are ordered from lowest to highest." if q["type"] == "score" else ""
    user = (f"{render(rec['state'])}\n\nQuestion: {render(q.get('instructions'))}{order}\n\nOptions:\n" + "\n".join(lines)
            + "\n\nReply with only the letter of the best option.")
    tok = teacher["tok"]
    msgs = [{"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok.encode(text, add_special_tokens=False)


def score(teacher, items, token_budget=4096, guard=None):
    """items: dicts with 'ids' and 'k'. -> list of option-logit lists (log-probs over the first k letters, renormalized)."""
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out = [None] * len(items)
    pad = teacher["tok"].pad_token_id or teacher["tok"].eos_token_id or 0
    i = 0
    while i < len(order):
        if guard is not None:
            guard.check()
        batch = [order[i]]
        while i + len(batch) < len(order) and len(items[order[i + len(batch)]]["ids"]) * (len(batch) + 1) <= token_budget:
            batch.append(order[i + len(batch)])
        L = max(len(items[j]["ids"]) for j in batch)
        ids = np.full((len(batch), L), pad, dtype=np.int32)
        last = np.zeros(len(batch), dtype=np.int32)
        for b, j in enumerate(batch):
            n = len(items[j]["ids"])
            ids[b, :n] = items[j]["ids"]
            last[b] = n - 1
        h = teacher["backbone"](mx.array(ids))
        h_last = h[mx.arange(len(batch)), mx.array(last)]
        logits = teacher["head"](h_last)[:, teacher["letter_ids"]].astype(mx.float32)
        mx.eval(logits)
        logits = np.array(logits)
        for b, j in enumerate(batch):
            z = logits[b, : items[j]["k"]]
            z = z - z.max()
            out[j] = (z - np.log(np.exp(z).sum())).tolist()
        i += len(batch)
    return out
