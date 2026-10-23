# tino

Turning small decoder LMs into [Jev](https://openrouter.ai/blog/insights/what-is-jev/)-style decision models: a state plus typed questions in (choice, score, yes/no), calibrated probabilities out, one forward pass, no generated text.

Three models, one per approach, all on the same base, data, and evaluation:

| Model | Approach | Status |
|---|---|---|
| [`tino-finetuned`](https://huggingface.co/BrandeisPatrick/tino-finetuned) | Method 0: full fine-tune of every transformer layer plus a pointer head (Adafactor, embeddings frozen) | done, run [`20261022-225150`](runs/20261022-225150-tino-finetuned) |
| `tino-adapter` | Method 1: rank-16 LoRA on a frozen base plus a pointer head | planned |
| `tino-distil` | Method 2: same student as the adapter, trained on Jev's probability distributions | planned |

## Setup

- **Base**: `Qwen/Qwen3-0.6B-Base`, pinned revision in each config.
- **Data**: [`jaredpalmer/kev-suites`](https://huggingface.co/datasets/jaredpalmer/kev-suites) pinned to one revision; every file is checked against its manifest checksum. Train on `decision-v7` (ten public datasets plus generated policy and rule records), select on its development split, fit temperature on its calibration split.
- **Evaluation**: `decision-v7` test (trained sources), `transfer-v4` dev and test (sources never trained on), `transfer-v9` test (adds MMLU-Pro, buried evidence, unknowable records). Test splits are read once per released model.
- **Reference point**: [Kev-0.6B](https://huggingface.co/jaredpalmer/kev-0.6b), a LoRA adapter on the same base and data: 0.808 trained-source and 0.642 new-source accuracy on its locked tests.
- **Hardware**: Apple M4, 16 GB unified memory, MLX. A resource guard (`tino/guard.py`) caps MLX memory and pauses or stops training when the Mac runs low, so a run never takes the machine down. Full fine-tuning with AdamW peaked at 10.1 GB, so the backbone uses Adafactor, whose optimizer state is a few MB instead of ~3.5 GB.

## Results

`tino-finetuned`, run [`20261022-225150-tino-finetuned`](runs/20261022-225150-tino-finetuned), checkpoint selected at step 2,520 of 2,881 by development accuracy. Test splits were read once. "Calibrated" divides logits by T = 2.45, fitted on the decision-v7 calibration split. Kev-0.6B numbers come from its model card (LoRA on the same base and data, evaluated with Kev's own harness), so treat the comparison as close, not exact.

| Suite / split | Questions | Accuracy | Kev-0.6B | Jev | Brier | ECE raw | ECE calibrated | Confident error, calibrated |
|---|---|---|---|---|---|---|---|---|
| decision-v7 dev (trained sources) | 1,468 | 0.794 | – | – | 0.307 | 0.103 | 0.031 | 2.6% |
| decision-v7 test (trained sources) | 1,440 | **0.810** | 0.808 | – | 0.295 | 0.105 | 0.038 | 2.6% |
| transfer-v4 dev (new sources) | 764 | 0.598 | 0.620 | 0.857 | 0.532 | 0.181 | 0.060 | 0.3% |
| transfer-v4 test (new sources) | 764 | 0.618 | 0.642 | – | 0.526 | 0.169 | 0.050 | 0.1% |
| transfer-v9 test (adds MMLU-Pro, buried, unknowable) | 1,264 | 0.544 | – | – | 0.622 | 0.194 | 0.041 | 0.2% |

What it shows:

- **Trained sources**: full fine-tuning matches the LoRA adapter (0.810 vs 0.808).
- **New sources**: it trails the adapter by about 2.5 points (0.618 vs 0.642). Updating every weight fits the training sources at least as well, but carries over slightly less to unseen ones.
- **Calibration**: raw probabilities are overconfident (ECE 0.10 in domain, 0.17 out of domain). One temperature brings ECE to 0.03–0.06 and confident errors to under 3%.
- **Weak spot**: Score questions (exact level on 5-point sentiment or star scales) reach only 0.51 on decision-v7 test, against 0.88 for Choice and 0.86 for yes/no.
- Per-source accuracy on transfer-v4 test: SciQ 0.85, QNLI 0.79, TweetEval-offensive 0.76, Emotion 0.56, PAWS 0.56, MMLU 0.40, held-out deadline rules 0.25.

Training cost on the M4: 183 minutes for 2 epochs (31,152 question passes), about 515 tokens per second, peak 7.7 GB of MLX memory, system free memory never below 31%. Inference: 66 ms per question in fp32.

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
