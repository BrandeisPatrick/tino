# tino

Turning small decoder LMs into [Jev](https://openrouter.ai/blog/insights/what-is-jev/)-style decision models: a state plus typed questions in (choice, score, yes/no), calibrated probabilities out, one forward pass, no generated text.

Three models, one per approach, all on the same base, data, and evaluation:

| Model | Approach | Status |
|---|---|---|
| `tino-finetuned` | Method 0: full fine-tune of every transformer layer plus a pointer head (Adafactor, embeddings frozen) | in progress |
| `tino-adapter` | Method 1: rank-16 LoRA on a frozen base plus a pointer head | planned |
| `tino-distil` | Method 2: same student as the adapter, trained on Jev's probability distributions | planned |

## Setup

- **Base**: `Qwen/Qwen3-0.6B-Base`, pinned revision in each config.
- **Data**: [`jaredpalmer/kev-suites`](https://huggingface.co/datasets/jaredpalmer/kev-suites) pinned to one revision; every file is checked against its manifest checksum. Train on `decision-v7` (ten public datasets plus generated policy and rule records), select on its development split, fit temperature on its calibration split.
- **Evaluation**: `decision-v7` test (trained sources), `transfer-v4` dev and test (sources never trained on), `transfer-v9` test (adds MMLU-Pro, buried evidence, unknowable records). Test splits are read once per released model.
- **Reference point**: [Kev-0.6B](https://huggingface.co/jaredpalmer/kev-0.6b), a LoRA adapter on the same base and data: 0.808 trained-source and 0.642 new-source accuracy on its locked tests.
- **Hardware**: Apple M4, 16 GB unified memory, MLX. A resource guard (`tino/guard.py`) caps MLX memory and pauses or stops training when the Mac runs low, so a run never takes the machine down. Full fine-tuning with AdamW peaked at 10.1 GB, so the backbone uses Adafactor, whose optimizer state is a few MB instead of ~3.5 GB.

## Tracking

Nothing lives only on the laptop.

- `runs/<run-id>/` holds each run's config, provenance (git commit, data revisions and checksums, library versions, hardware), training log, development evals, and final results. The training script commits and pushes this directory at every evaluation.
- Weights go to private Hugging Face Hub repos named after the model.
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
