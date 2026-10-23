"""Decoder backbone + pointer head: one forward pass, a probability distribution over the options of each question."""
import importlib
import json
import math
import os
import shutil

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten


class PointerHead(nn.Module):
    """score(option) = <W_k h(</OPT>), W_q h(<DECIDE>)> / sqrt(dp)."""

    def __init__(self, hidden, dp=256):
        super().__init__()
        self.q = nn.Linear(hidden, dp)
        self.k = nn.Linear(hidden, dp)
        self.scale = 1 / math.sqrt(dp)

    def __call__(self, h_decide, h_opts):  # [B,d], [B,K,d] -> [B,K]
        return (self.k(h_opts) @ self.q(h_decide)[..., None]).squeeze(-1) * self.scale


class DecisionModel(nn.Module):
    def __init__(self, backbone, hidden, dp=256):
        super().__init__()
        self.backbone = backbone
        self.head = PointerHead(hidden, dp)

    def __call__(self, ids, decide_idx, opt_idx, opt_mask):
        h = self.backbone(ids)
        B, L, D = h.shape
        flat = h.reshape(B * L, D)
        base = mx.arange(B) * L
        h_dec = flat[base + decide_idx]
        h_opt = flat[(base[:, None] + opt_idx).reshape(-1)].reshape(B, opt_idx.shape[1], D)
        z = self.head(h_dec.astype(mx.float32), h_opt.astype(mx.float32))
        return mx.where(opt_mask, z, mx.array(-1e9, dtype=z.dtype))


def collate(batch, pad_id):
    L = max(len(r["ids"]) for r in batch)
    K = max(len(r["opt_idx"]) for r in batch)
    ids = np.full((len(batch), L), pad_id, dtype=np.int32)
    opt = np.zeros((len(batch), K), dtype=np.int32)
    mask = np.zeros((len(batch), K), dtype=bool)
    for b, r in enumerate(batch):
        ids[b, : len(r["ids"])] = r["ids"]
        opt[b, : len(r["opt_idx"])] = r["opt_idx"]
        mask[b, : len(r["opt_idx"])] = True
    dec = np.array([r["decide_idx"] for r in batch], dtype=np.int32)
    lab = np.array([r["label"] if r.get("label") is not None else 0 for r in batch], dtype=np.int32)
    return mx.array(ids), mx.array(dec), mx.array(opt), mx.array(mask), mx.array(lab)


def _backbone_from_config(config):
    arch = importlib.import_module(f"mlx_lm.models.{config['model_type']}")
    return arch.Model(arch.ModelArgs.from_dict(config)).model


def load_base(repo, revision=None, dp=256):
    """-> (DecisionModel with pretrained backbone and a fresh head, HF tokenizer, base config)."""
    from mlx_lm import load
    lm, tok, config = load(repo, revision=revision, return_config=True)
    return DecisionModel(lm.model, config["hidden_size"], dp), tok._tokenizer, config


def save_checkpoint(model, out_dir, tokenizer, base_config, meta, dtype=None):
    """Self-contained directory: weights, base config (architecture), tokenizer, and tino metadata."""
    os.makedirs(out_dir, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    if dtype is not None:
        weights = {k: (v.astype(dtype) if k.startswith("backbone.") else v.astype(mx.float32)) for k, v in weights.items()}
    mx.save_safetensors(os.path.join(out_dir, "model.safetensors"), weights)
    json.dump(base_config, open(os.path.join(out_dir, "config.json"), "w"), indent=1)
    json.dump(meta, open(os.path.join(out_dir, "tino_config.json"), "w"), indent=1)
    tokenizer.save_pretrained(out_dir)


def load_checkpoint(path_or_repo, revision=None):
    """Load a saved tino checkpoint from a local directory or a Hub repo id."""
    from transformers import AutoTokenizer
    path = path_or_repo
    if not os.path.isdir(path):
        from huggingface_hub import snapshot_download
        path = snapshot_download(path_or_repo, revision=revision)
    config = json.load(open(os.path.join(path, "config.json")))
    meta = json.load(open(os.path.join(path, "tino_config.json")))
    model = DecisionModel(_backbone_from_config(config), config["hidden_size"], meta.get("pointer_dim", 256))
    model.load_weights(os.path.join(path, "model.safetensors"))
    mx.eval(model.parameters())
    return model, AutoTokenizer.from_pretrained(path), meta


def copy_checkpoint(src, dst):
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
