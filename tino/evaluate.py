"""Batched inference over encoded rows."""
import time

import mlx.core as mx
import numpy as np

from .data import batches
from .model import collate


def predict(model, rows, pad_id, token_budget=4096, guard=None):
    """-> list of prediction dicts (logits, label, src, qtype, ids) in the order of rows, plus seconds per question."""
    out = [None] * len(rows)
    t0 = time.time()
    for idx in batches(rows, token_budget, shuffle=False):
        if guard is not None:
            guard.check()
        batch = [rows[i] for i in idx]
        ids, dec, opt, mask, _ = collate(batch, pad_id)
        z = model(ids, dec, opt, mask)
        mx.eval(z)
        z = np.array(z)
        for j, i in enumerate(idx):
            r = rows[i]
            out[i] = {"logits": z[j, : len(r["opt_idx"])].tolist(), "label": r["label"], "src": r["src"],
                      "qtype": r["qtype"], "rec_id": r["rec_id"], "rec_idx": r.get("rec_idx"), "qid": r["qid"], "keys": r["keys"]}
    return out, (time.time() - t0) / max(len(rows), 1)
