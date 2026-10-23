"""Render Jev / TypeSafe-shaped requests into token rows for a decoder LM.

One causal row per question:

    <STATE> state tokens  <Q> instruction tokens  (<OPT> option tokens </OPT>)*  <DECIDE>

The pointer head reads the hidden state at every </OPT> and at <DECIDE>. Rendering of states, option text, keys and
labels follows Kev's conventions (github.com/jaredpalmer/kev, Apache-2.0) so kev-suites records mean the same thing
here as they do there. Delimiters reuse rarely used Qwen special tokens, so no embedding rows are added.
"""
import re

SPECIAL = {
    "state": "<|fim_prefix|>",
    "q": "<|fim_middle|>",
    "opt": "<|box_start|>",
    "end_opt": "<|box_end|>",
    "decide": "<|fim_suffix|>",
}
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def render(v, indent=0):
    """Flatten str | object | array into the text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
        for k, x in v.items()
    )


def option_text(name, desc):
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def question_options(q):
    """-> (keys, option texts). Keys are what probabilities are reported under, in option order."""
    t, c = q["type"], q.get("criteria")
    if t == "noul":
        c = c or {}
        return ["false", "true"], [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
    if t == "choice":
        return list(c), [option_text(k, v) for k, v in c.items()]
    if t == "score":
        return [str(i) for i in range(len(c))], [render(x) for x in c]
    raise ValueError(f"unknown question type {t!r}")


def label_index(q, keys):
    """Gold label as an option index, or None when the record carries no label."""
    y = q.get("label")
    if y is None:
        return None
    if q["type"] == "noul":
        return 1 if y else 0
    if q["type"] == "choice":
        return keys.index(y)
    return int(y)


class Encoder:
    def __init__(self, tokenizer, max_state=384, max_row=1536):
        self.tok = tokenizer
        self.max_state, self.max_row = max_state, max_row
        self.sid = {k: tokenizer.convert_tokens_to_ids(v) for k, v in SPECIAL.items()}
        if any(i is None or i == tokenizer.unk_token_id for i in self.sid.values()):
            raise ValueError(f"tokenizer lacks delimiter tokens: {self.sid}")
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def user_tokens(self, text):
        """Caller text can never produce delimiter tokens: '<|x|>' is rewritten before tokenizing."""
        return self.tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids

    def state_ids(self, state):
        ids = self.user_tokens(render(state))
        return [self.sid["state"]] + ids[: self.max_state - 1], len(ids) + 1 > self.max_state

    def question_row(self, state_ids, q):
        keys, opts = question_options(q)
        ids = list(state_ids) + [self.sid["q"]] + self.user_tokens(render(q.get("instructions")))
        opt_idx = []
        for o in opts:
            ids += [self.sid["opt"]] + self.user_tokens(o) + [self.sid["end_opt"]]
            opt_idx.append(len(ids) - 1)
        ids.append(self.sid["decide"])
        return {"ids": ids, "opt_idx": opt_idx, "decide_idx": len(ids) - 1, "keys": keys}

    def encode_record(self, rec, rec_id=None):
        """-> list of rows, one per question, each with label index, type and source attached."""
        s_ids, truncated = self.state_ids(rec["state"])
        rows = []
        for qid, q in rec["questions"].items():
            r = self.question_row(s_ids, q)
            r.update(qid=qid, rec_id=rec_id, qtype=q["type"], label=label_index(q, r["keys"]),
                     src=q.get("src") or (rec.get("_meta") or {}).get("source"), state_truncated=truncated,
                     variant=(rec.get("_meta") or {}).get("variant"))
            rows.append(r)
        return rows
