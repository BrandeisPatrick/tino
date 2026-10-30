#!/usr/bin/env python
"""Port Kev-0.6B's published adapter and head into a tino checkpoint directory, so the reference model is scored by
this harness on the same splits. Kev's LoRA (PEFT, rank 16, alpha 32, the same seven projections) and its pointer
head (two 1024 -> 256 projections) match tino's architecture exactly, so the port is a renaming plus a transpose.
PyTorch is not needed: head.pt is read with a minimal unpickler.

    uv run python scripts/kev_port.py --out checkpoints/kev-0.6b

Validate against Kev's model card afterwards (scripts/eval_suites.py does this): decision-v4 development 0.801,
transfer-v4 development 0.620.
"""
import argparse
import collections
import hashlib
import io
import json
import os
import pickle
import re
import shutil
import zipfile

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from safetensors.numpy import load_file, save_file

REPO = "jaredpalmer/kev-0.6b"
PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                   "added_tokens.json"]


class TorchFree(pickle.Unpickler):
    """Enough of torch's pickle protocol to read a state dict of dense tensors."""

    def __init__(self, f, zf, prefix):
        super().__init__(f)
        self.zf, self.prefix = zf, prefix

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            def rebuild(storage, offset, size, stride, *rest):
                it = storage.dtype.itemsize
                return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                                       strides=tuple(s * it for s in stride)).copy()
            return rebuild
        if name == "_rebuild_parameter":
            return lambda data, *rest: data
        if module.startswith("torch") and name.endswith("Storage"):
            return {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
                    "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_, "BFloat16Storage": "bf16"}[name]
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"unsupported {module}.{name}")

    def persistent_load(self, pid):
        _, storage_type, key, _, _ = pid
        raw = self.zf.read(f"{self.prefix}/data/{key}")
        if storage_type == "bf16":
            return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        return np.frombuffer(raw, dtype=storage_type)


def read_pt(path):
    zf = zipfile.ZipFile(path)
    pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    return TorchFree(io.BytesIO(zf.read(pkl)), zf, pkl[: -len("/data.pkl")]).load()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/kev-0.6b")
    ap.add_argument("--revision", default=None, help="Hub revision of the Kev repo (default: current)")
    args = ap.parse_args()
    rev = args.revision or HfApi().model_info(REPO).sha
    get = lambda f: hf_hub_download(REPO, f, revision=rev)  # noqa: E731

    cfg = json.load(open(get("adapter_config.json")))
    assert cfg["peft_type"] == "LORA" and sorted(cfg["target_modules"]) == sorted(PROJECTIONS), cfg
    assert not cfg.get("use_rslora") and not cfg.get("use_dora") and cfg.get("modules_to_save") in (None, []), cfg
    head = read_pt(get("head.pt"))
    assert not head.get("special_embeddings") and not head.get("option_isolation"), head
    adapter = load_file(get("adapter_model.safetensors"))

    weights = {}
    for k, v in adapter.items():
        m = re.match(r"base_model\.model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$", k)
        assert m, k
        # PEFT keeps A as [r, in] and B as [out, r]; mlx_lm keeps lora_a as [in, r] and lora_b as [r, out].
        weights[f"backbone.layers.{m.group(1)}.{m.group(2)}.lora_{m.group(3).lower()}"] = np.ascontiguousarray(v.T.astype(np.float32))
    for k, v in head["head"].items():
        weights[f"head.{k}"] = np.ascontiguousarray(v.astype(np.float32))
    expected = 2 * len(PROJECTIONS) * 28 + 4
    assert len(weights) == expected, (len(weights), expected)

    out = args.out
    os.makedirs(out, exist_ok=True)
    save_file(weights, os.path.join(out, "model.safetensors"))
    shutil.copy(hf_hub_download(head["base"], "config.json", revision=head["base_revision"]), os.path.join(out, "config.json"))
    for f in TOKENIZER_FILES:
        shutil.copy(get(f), os.path.join(out, f))
    meta = {
        "name": "kev-0.6b", "method": "lora", "base_model": head["base"], "base_revision": head["base_revision"],
        "pointer_dim": int(head["head_dim"]), "max_state": 384, "max_row": 1536, "temperature": 1.0, "weights": "adapter",
        "lora": {"rank": cfg["r"], "scale": cfg["lora_alpha"] / cfg["r"], "dropout": cfg["lora_dropout"]},
        "base_dtype": "float32",
        "source": {"repo": REPO, "revision": rev, "trained_dtype": (head.get("args") or {}).get("dtype"),
                   "files": {f: sha256(get(f)) for f in ["adapter_model.safetensors", "head.pt", "adapter_config.json"]},
                   "kev_training_args": head.get("args")},
    }
    json.dump(meta, open(os.path.join(out, "tino_config.json"), "w"), indent=1)
    print(f"[kev_port] {len(weights)} tensors -> {out}; base {head['base']}@{head['base_revision'][:8]}, "
          f"LoRA r{cfg['r']} scale {meta['lora']['scale']}, head dim {head['head_dim']}, Kev revision {rev[:8]}")


if __name__ == "__main__":
    main()
