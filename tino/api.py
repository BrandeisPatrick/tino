"""Answer Jev-shaped requests with a tino checkpoint.

    from tino.api import Decider
    d = Decider.load("BrandeisPatrick/tino-finetuned")
    d.decide({"state": "...", "questions": {"urgent": {"type": "noul", "instructions": "Needs a reply within an hour?"}}})

Output follows TypeSafe's /v1/systemone answer shapes: noul -> probability of yes; choice -> most likely key, per-key
probabilities, confidence; score -> expected level, legend, per-level probabilities, confidence. Confidence formulas
match the ones Kev documents from TypeSafe's reference adapter.
"""
import mlx.core as mx
import numpy as np

from .format import Encoder, question_options, render
from .metrics import softmax
from .model import collate, load_checkpoint


def choice_confidence(p):
    K = len(p)
    return 1.0 if K == 1 else float((max(p) - 1 / K) / (1 - 1 / K))


def score_confidence(p):
    L = len(p)
    if L == 1:
        return 1.0
    mode = int(np.argmax(p))
    D = sum(abs(i - (L - 1) / 2) for i in range(L)) / L
    return float(max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / D))


class Decider:
    def __init__(self, model, tokenizer, meta):
        self.model, self.meta = model, meta
        self.enc = Encoder(tokenizer, meta.get("max_state", 384) * 16, meta.get("max_row", 1536) * 16)
        self.temperature = meta.get("temperature", 1.0)
        model.eval()

    @classmethod
    def load(cls, path_or_repo, revision=None):
        return cls(*load_checkpoint(path_or_repo, revision))

    def decide(self, request, calibrated=True):
        s_ids, truncated = self.enc.state_ids(request["state"])
        qids = list(request["questions"])
        rows = [self.enc.question_row(s_ids, request["questions"][q]) for q in qids]
        ids, dec, opt, mask, _ = collate(rows, self.enc.pad_id)
        z = np.array(self.model(ids, dec, opt, mask))
        T = self.temperature if calibrated else 1.0
        answers = {}
        for j, qid in enumerate(qids):
            q = request["questions"][qid]
            keys, opts = question_options(q)
            p = softmax(z[j, : len(keys)], T)
            if q["type"] == "noul":
                answers[qid] = {"type": "noul", "noul": round(float(p[1]), 4)}
            elif q["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": keys[int(np.argmax(p))], "confidence": round(choice_confidence(p), 4),
                                "probabilities": {k: round(float(v), 4) for k, v in zip(keys, p)}}
            else:
                answers[qid] = {"type": "score", "score": round(float(sum(i * v for i, v in enumerate(p))), 4),
                                "legend": dict(zip(keys, opts)), "probabilities": {k: round(float(v), 4) for k, v in zip(keys, p)},
                                "confidence": round(score_confidence(p), 4)}
        return {"answers": answers, "state_truncated": truncated}
