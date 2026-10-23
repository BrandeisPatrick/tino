"""Distillation targets and loss.

Teacher log-probs (tino/teacher.py, raw T=1) are softened with the teacher temperature fitted on the calibration split,
then the student minimizes  alpha * KL(teacher || student) + (1 - alpha) * CE(gold)  per question. Questions without a
teacher distribution (more options than letters) use CE alone.
"""
import json

import mlx.core as mx
import numpy as np
from huggingface_hub import hf_hub_download


def load_teacher_labels(repo, prefix, revision, suite, split):
    man = json.load(open(hf_hub_download(repo, f"{prefix}/manifest.json", repo_type="dataset", revision=revision)))
    path = hf_hub_download(repo, f"{prefix}/{suite}/{split}.jsonl", repo_type="dataset", revision=revision)
    labels = {}
    for line in open(path):
        r = json.loads(line)
        labels[(r["rec_idx"], r["qid"])] = r["teacher_logp"]
    return labels, man


def attach(rows, labels, temperature):
    """Adds row['teacher_p'] (np.array over options, tempered) or None. Returns coverage counts."""
    hit = 0
    for r in rows:
        lp = labels.get((r["rec_idx"], r["qid"]))
        if lp is None or len(lp) != len(r["opt_idx"]):
            r["teacher_p"] = None
            continue
        z = np.asarray(lp, dtype=np.float64) / temperature
        z -= z.max()
        p = np.exp(z)
        r["teacher_p"] = p / p.sum()
        hit += 1
    return {"with_teacher": hit, "without_teacher": len(rows) - hit}


def collate_teacher(batch, K):
    P = np.zeros((len(batch), K), dtype=np.float32)
    has = np.zeros(len(batch), dtype=np.float32)
    for b, r in enumerate(batch):
        if r.get("teacher_p") is not None:
            P[b, : len(r["teacher_p"])] = r["teacher_p"]
            has[b] = 1.0
    return mx.array(P), mx.array(has)


def distil_loss(z, labels, teacher_p, has_teacher, alpha):
    """z: student logits [B,K] (padded options at -1e9). Returns per-row loss [B]."""
    logp = z - mx.logsumexp(z, axis=-1, keepdims=True)
    ce = -mx.take_along_axis(logp, labels[:, None], axis=-1).squeeze(-1)
    safe = mx.where(teacher_p > 0, teacher_p, mx.ones_like(teacher_p))
    kl = (teacher_p * (mx.log(safe) - logp)).sum(axis=-1)
    return has_teacher * (alpha * kl + (1 - alpha) * ce) + (1 - has_teacher) * ce
