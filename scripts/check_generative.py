"""CPU checks for the generative arm (tino/generative.py) on a tiny random Qwen3 built from the real tokenizer.
Each fast path is compared with a slow, obvious one that recomputes the whole sequence with no cache: answer-position
loss, cached option scores, cached greedy answers, and one combined pass against separate passes; plus row parity
with the decision encoder and the answer parser.

The references project to the vocabulary only at the positions they need, so the checks stay under about 1 GB. The
first version built full-vocabulary logits (several GB) and, run beside the thinking evaluation on October 31, pushed
swap past that job's guard limit: still run this only while the queue is held at a job boundary (queue/hold).

    uv run python scripts/check_generative.py
"""
import glob, json, os, sys, tempfile
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, W)
import mlx.core as mx
mx.set_default_device(mx.cpu)
import mlx.nn as nn
import mlx.utils
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3
from tino import generative as gen
from tino.data import build_rows, load_records
from tino.format import Encoder
from tino.model import apply_lora


def make_tiny(out):
    """2 layers, hidden 64, tied embeddings, random weights; tokenizer files from the pinned Qwen3-0.6B-Base."""
    from huggingface_hub import snapshot_download
    src = snapshot_download("Qwen/Qwen3-0.6B-Base", revision="da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
                            allow_patterns=["config.json", "tokenizer*", "vocab.json", "merges.txt"])
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
               head_dim=16, tie_word_embeddings=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"))
    for f in glob.glob(os.path.join(src, "*")):
        if os.path.basename(f) != "config.json":
            with open(f, "rb") as a, open(os.path.join(out, os.path.basename(f)), "wb") as b:
                b.write(a.read())
    mx.random.seed(0)
    mx.save_safetensors(os.path.join(out, "model.safetensors"), dict(tree_flatten(qwen3.Model(qwen3.ModelArgs.from_dict(cfg)).parameters())))


TINY = tempfile.mkdtemp(prefix="tino-tiny-")
make_tiny(TINY)

fails = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)
    if not ok: fails.append(name)

lm, tokw, cfg = gen.load_lm(TINY)
tok = tokw._tokenizer
recs, _ = load_records("decision-v7", "development")
dec_rows, _ = build_rows(Encoder(tok), recs[:400], "decision-v7")
enc = gen.GenEncoder(tok, mode="text")
rows, _ = build_rows(enc, recs[:400], "decision-v7")
check("text-mode rows identical to decision rows", len(rows) == len(dec_rows) and all(a["ids"] == b["ids"] and a["opt_idx"] == b["opt_idx"] for a, b in zip(rows, dec_rows)), f"{len(rows)} rows")
check("answer ids end with eos and re-decode to option text", all(a[-1] == tok.eos_token_id and tok.decode(a[:-1]) == t for r in rows for a, t in zip(r["answer_ids"], r["answers"])))
lrows, _ = build_rows(gen.GenEncoder(tok, mode="letter"), recs[:400], "decision-v7")
check("letter mode answers", all(r["answers"][:2] == ["A", "B"] for r in lrows) and gen.letter(26) == "AA" and gen.letter(77) == "BZ")

model = gen.GenModel(lm)
apply_lora(model, {"rank": 4, "scale": 2.0, "dropout": 0.0})
gen.freeze_output(model)
# random LoRA weights (B starts at zero), so the adapter changes the outputs these checks compare
params = model.trainable_parameters()
params = mlx.utils.tree_map(lambda x: mx.random.normal(x.shape) * 0.05, params)
model.update(params)
mx.eval(model.parameters())
check("only LoRA is trainable", all("lora" in k for k, _ in tree_flatten(model.trainable_parameters())), f"{sum(v.size for _, v in tree_flatten(model.trainable_parameters()))} params")
model.eval()

# loss at answer positions == loss from full logits
batch = [r for r in rows if len(r["ids"]) < 300][:6]
ids, pos, tgt, own = gen.collate_gen(batch, enc.pad_id)
loss, correct = gen.gen_loss(model, ids, pos, tgt, own, len(batch))
def logp_at(h, t, tok_id):
    """log p(tok_id) from hidden state h[t], projected alone."""
    z = model.project(h[t][None])[0]
    return (z[tok_id] - mx.logsumexp(z)).item()

hb = lm.model(ids)
ref = 0.0
for b, r in enumerate(batch):
    s = r["ids"] + r["answer_ids"][r["label"]]
    for t in range(len(r["ids"]), len(s)):
        ref -= logp_at(hb[b], t - 1, s[t])
check("answer-position loss equals full-logit loss", abs(loss.item() - ref) < 1e-3 * max(1, abs(ref)), f"{loss.item():.5f} vs {ref:.5f}")

# gradients flow, with gradient checkpointing on
model.train()
gen.grad_checkpoint(model.backbone.layers[0])
lg = nn.value_and_grad(model, gen.gen_loss)
(l2, _), grads = lg(model, ids, pos, tgt, own, len(batch))
mx.eval(l2, grads)
gn = sum(float((v * v).sum()) for _, v in tree_flatten(grads)) ** 0.5
check("checkpointed training step gives finite loss and non-zero gradients", np.isfinite(l2.item()) and gn > 0, f"loss {l2.item():.3f} grad norm {gn:.4f}")
model.eval()

def naive_seq_logprob(r, a):
    s = r["ids"] + a
    h = lm.model(mx.array(s)[None])[0]
    return sum(logp_at(h, t - 1, s[t]) for t in range(len(r["ids"]), len(s)))

many = [r for r in rows if len(r["answer_ids"]) > 8][:2]
probe = [r for r in rows if len(r["answer_ids"]) <= 8][:4] + many
with gen.no_checkpoint():
    sc = gen.read_rows(model, lm, tokw, probe, write=False, score=True, chunk=4)["scores"]
diffs = [abs(s - naive_seq_logprob(r, a)) for r, ss in zip(probe, sc) for a, s in zip(r["answer_ids"], ss)]
check("cached option scores equal full-sequence log-probs (incl. >4 options, several chunks)", max(diffs) < 2e-3, f"max diff {max(diffs):.2e} over {len(diffs)} options; option counts {[len(r['answer_ids']) for r in probe]}")

def naive_greedy(r, max_new):
    s, out = list(r["ids"]), []
    for _ in range(max_new):
        z = model.project(lm.model(mx.array(s)[None])[0, -1][None])[0]
        t = int(mx.argmax(z).item())
        if t in tokw.eos_token_ids:
            break
        out.append(t); s.append(t)
    return out

gr = rows[:12]
with gen.no_checkpoint():
    got = gen.read_rows(model, lm, tokw, gr, write=True, score=False)["written"]
same = [g == naive_greedy(r, max(len(a) for a in r["answer_ids"]) + 1) for r, g in zip(gr, got)]
check("cached greedy answers equal uncached greedy answers", all(same), f"{sum(same)}/{len(same)} identical")
both_rows = gr[:6] + many[:1]
with gen.no_checkpoint():
    both = gen.read_rows(model, lm, tokw, both_rows, write=True, score=True, chunk=4)
    w_only = gen.read_rows(model, lm, tokw, both_rows, write=True, score=False)["written"]
    s_only = gen.read_rows(model, lm, tokw, both_rows, write=False, score=True, chunk=4)["scores"]
sd = max(abs(a - b) for x, y in zip(both["scores"], s_only) for a, b in zip(x, y))
check("one pass writing and scoring equals separate passes", both["written"] == w_only and sd < 1e-5, f"score diff {sd:.1e}")

r0 = rows[0]
check("parse: exact tokens", gen.parse_written(r0, r0["answer_ids"][1][:-1], tok, tok.eos_token_id) == 1)
check("parse: case, spacing, trailing period", gen.parse_written(r0, tok.encode("  " + r0["answers"][0].upper() + ". "), tok) == 0)
rc = next(r for r in rows if ": " in r["answers"][0])
check("parse: option name alone", gen.parse_written(rc, tok.encode(rc["answers"][1].split(": ")[0]), tok) == 1, rc["answers"][:2])
check("parse: garbage is invalid", gen.parse_written(r0, tok.encode("the answer is unclear"), tok) == -1)
print("ALL PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
