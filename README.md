# tino

Turning small decoder LMs into [Jev](https://openrouter.ai/blog/insights/what-is-jev/)-style decision models: a state plus typed questions in (choice, score, yes/no), calibrated probabilities out, one forward pass, no generated text.

Three models, one per approach, all on the same base, data, and evaluation:

| Model | Approach | Status |
|---|---|---|
| [`tino-finetuned`](https://huggingface.co/BrandeisPatrick/tino-finetuned) | Method 0: full fine-tune of every transformer layer plus a pointer head (Adafactor, embeddings frozen) | done, run [`20261022-225150`](runs/20261022-225150-tino-finetuned) |
| [`tino-adapter`](https://huggingface.co/BrandeisPatrick/tino-adapter) | Method 1: rank-16 LoRA on a frozen base plus a pointer head (Kev's LoRA recipe) | done, run [`20261023-085627`](runs/20261023-085627-tino-adapter) |
| `tino-distil` | Method 2: same student as the adapter, trained on Jev's probability distributions | planned |

## Setup

- **Base**: `Qwen/Qwen3-0.6B-Base`, pinned revision in each config.
- **Data**: [`jaredpalmer/kev-suites`](https://huggingface.co/datasets/jaredpalmer/kev-suites) pinned to one revision; every file is checked against its manifest checksum. Train on `decision-v7` (ten public datasets plus generated policy and rule records), select on its development split, fit temperature on its calibration split.
- **Evaluation**: `decision-v7` test (trained sources), `transfer-v4` dev and test (sources never trained on), `transfer-v9` test (adds MMLU-Pro, buried evidence, unknowable records). Test splits are read once per released model.
- **Reference point**: [Kev-0.6B](https://huggingface.co/jaredpalmer/kev-0.6b), a LoRA adapter on the same base and data: 0.808 trained-source and 0.642 new-source accuracy on its locked tests.
- **Hardware**: Apple M4, 16 GB unified memory, MLX. A resource guard (`tino/guard.py`) caps MLX memory and pauses or stops training when the Mac runs low, so a run never takes the machine down. Full fine-tuning with AdamW peaked at 10.1 GB, so the backbone uses Adafactor, whose optimizer state is a few MB instead of ~3.5 GB.

## Results

Both models share the base, data, pointer head, schedule, batching, and fp32 numerics; only how the backbone trains differs. Each checkpoint was selected by development accuracy, test splits were read once, and "calibrated" divides logits by a temperature fitted on the decision-v7 calibration split. Kev-0.6B numbers come from its model card (LoRA on the same base and data, evaluated with Kev's own harness), so that column is close, not exact.

**Accuracy**

| Suite / split | Questions | tino-finetuned | tino-adapter | Kev-0.6B | Jev |
|---|---|---|---|---|---|
| decision-v7 dev (trained sources) | 1,468 | 0.794 | 0.788 | – | – |
| decision-v7 test (trained sources) | 1,440 | **0.810** | **0.810** | 0.808 | – |
| transfer-v4 dev (new sources) | 764 | 0.598 | 0.594 | 0.620 | 0.857 |
| transfer-v4 test (new sources) | 764 | 0.618 | 0.613 | 0.642 | – |
| transfer-v9 test (adds MMLU-Pro, buried, unknowable) | 1,264 | 0.544 | 0.548 | – | – |

**Calibration** (test splits; ECE and confident errors after temperature scaling)

| Suite / split | Brier, finetuned | Brier, adapter | ECE, finetuned | ECE, adapter | Confident error, finetuned | Confident error, adapter |
|---|---|---|---|---|---|---|
| decision-v7 test | 0.295 | 0.286 | 0.037 | 0.034 | 2.6% | 1.7% |
| transfer-v4 test | 0.526 | 0.529 | 0.050 | 0.055 | 0.1% | 1.4% |
| transfer-v9 test | 0.622 | 0.622 | 0.041 | 0.071 | 0.2% | 1.7% |

**Cost on the M4**

| | tino-finetuned | tino-adapter |
|---|---|---|
| Trainable parameters | 441M | 10.6M (2.4% of that) |
| Published weights | 2.1 GB (full model) | 43 MB (adapter + head; base from the Hub) |
| Training time, 2 epochs | 183 min | 193 min |
| Peak MLX memory | 7.7 GB* | 4.0 GB |
| Lowest system free memory | 31% | 58% |
| Fitted temperature | 2.45 | 2.01 |
| Best checkpoint | step 2,520 of 2,881 | step 2,881 of 2,881 |
| Inference, fp32 | 40–66 ms per question | 52–85 ms per question (LoRA unfused) |

\* Measured before a fix that evaluates loss and gradients together. Reading the loss first had kept every activation alive and defeated gradient checkpointing. The fix changes memory only; results are identical.

What it shows:

- **Same accuracy, 40x fewer trainable weights.** On this base and data, full fine-tuning and LoRA land within half a point of each other on every suite. The adapter is 43 MB instead of 2.1 GB and trains in 4 GB of memory.
- **No transfer advantage for LoRA here.** Both trail Kev-0.6B by about 2.5 points on new sources. Since the two tino models agree with each other, the gap more likely comes from recipe or harness differences than from the method.
- **Calibration is good after one temperature.** Raw probabilities are overconfident for both (ECE about 0.10 in domain, 0.15–0.19 out of domain). Temperature scaling brings ECE to 0.03–0.07.
- **Score questions remain the weak spot.** Exact-level accuracy on 5-point scales is 0.51 (finetuned) and 0.58 (adapter), against 0.85–0.88 for Choice and yes/no.
- **The adapter was still improving at the last step**, so a third epoch might help it; the full fine-tune peaked before the end.

## Tracking

Nothing lives only on the laptop.

- `runs/<run-id>/` holds each run's config, provenance (git commit, data revisions and checksums, library versions, hardware), training log, development evals, and final results. The training script commits and pushes this directory at every evaluation.
- Weights go to private Hugging Face Hub repos named after the model; each run's `checkpoint.json` records the exact Hub revision.
- Weights & Biases is used automatically when the machine is logged in.
- Training refuses to start from uncommitted code, so every run maps to a commit.

## Usage

```bash
uv sync
uv run python scripts/train.py --config configs/tino-finetuned.json --smoke   # end-to-end check on a tiny subset
uv run python scripts/train.py --config configs/tino-finetuned.json
```

## Credits

The input layout, delimiter tokens, rendering rules, and pointer head follow [Kev](https://github.com/jaredpalmer/kev) (Apache-2.0), reimplemented for MLX.
