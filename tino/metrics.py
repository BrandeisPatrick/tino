"""Accuracy and calibration metrics over per-question predictions."""
from collections import defaultdict

import numpy as np


def softmax(z, T=1.0):
    z = np.asarray(z, dtype=np.float64) / T
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def summarize(preds, T=1.0, bins=15):
    """preds: dicts with 'logits' (list), 'label' (int), 'src', 'qtype'. Brier is the multi-class sum of squares."""
    if not preds:
        return {"n": 0}
    conf, correct, nll, brier = [], [], [], []
    by_src, by_type = defaultdict(list), defaultdict(list)
    for p in preds:
        pr = softmax(p["logits"], T)
        y = p["label"]
        ok = int(np.argmax(pr) == y)
        onehot = np.zeros_like(pr)
        onehot[y] = 1
        conf.append(pr.max()); correct.append(ok)
        nll.append(-np.log(max(pr[y], 1e-12))); brier.append(float(((pr - onehot) ** 2).sum()))
        by_src[p.get("src") or "?"].append(ok); by_type[p["qtype"]].append(ok)
    conf, correct = np.array(conf), np.array(correct)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return {
        "n": len(preds),
        "accuracy": float(correct.mean()),
        "nll": float(np.mean(nll)),
        "brier": float(np.mean(brier)),
        "ece": float(ece),
        "confident_error": float(((conf >= 0.9) & (correct == 0)).mean()),
        "mean_confidence": float(conf.mean()),
        "macro_source_accuracy": float(np.mean([np.mean(v) for v in by_src.values()])),
        "by_source": {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(by_src.items())},
        "by_type": {k: {"n": len(v), "accuracy": float(np.mean(v))} for k, v in sorted(by_type.items())},
    }


def fit_temperature(preds):
    """Single temperature minimizing NLL: coarse log grid, then golden-section refinement."""
    def nll(T):
        return float(np.mean([-np.log(max(softmax(p["logits"], T)[p["label"]], 1e-12)) for p in preds]))
    grid = np.exp(np.linspace(np.log(0.1), np.log(25.0), 61))
    i = int(np.argmin([nll(t) for t in grid]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (5 ** 0.5 - 1) / 2
    for _ in range(30):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return float((lo + hi) / 2)
