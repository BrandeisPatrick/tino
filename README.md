# tino

Turning small decoder LMs into [Jev](https://openrouter.ai/blog/insights/what-is-jev/)-style decision models: a state plus typed questions in (choice, score, yes/no), calibrated probabilities out, one forward pass, no generated text.

The research questions, experiments, and answers are in [RESEARCH.md](RESEARCH.md).

Five released models, all on the same 0.6B base family, data, and evaluation:

| Model | Approach | Trained sources | New sources |
|---|---|---|---|
| [`tino-finetuned`](https://huggingface.co/BrandeisPatrick/tino-finetuned) | Full fine-tune of every transformer layer plus a pointer head (Adafactor, embeddings frozen) | 0.810 | 0.618 |
| [`tino-adapter`](https://huggingface.co/BrandeisPatrick/tino-adapter) | Rank-16 LoRA on a frozen base plus a pointer head (Kev's LoRA recipe) | 0.810 | 0.613 |
| [`tino-adapter-post`](https://huggingface.co/BrandeisPatrick/tino-adapter-post) | Same adapter, starting from post-trained Qwen3-0.6B instead of the base | 0.810 | 0.615 |
| [`tino-distil`](https://huggingface.co/BrandeisPatrick/tino-distil) | Same adapter, trained half on gold labels and half on an autoregressive Qwen3-4B teacher's option probabilities; calibrated without temperature scaling | 0.808 | 0.636 |
| [`tino-distil-a1`](https://huggingface.co/BrandeisPatrick/tino-distil-a1) | Same adapter, trained on the Qwen3-4B teacher only, no gold labels; best out of domain | 0.750 | 0.671 |

Test accuracy on `decision-v7` (trained sources) and `transfer-v4` (new sources). For reference, Kev-0.6B scores 0.808 / 0.642 and Jev 0.857 on new-source development data.

## Setup

- **Base**: `Qwen/Qwen3-0.6B-Base`, pinned revision in each config.
- **Data**: [`jaredpalmer/kev-suites`](https://huggingface.co/datasets/jaredpalmer/kev-suites) pinned to one revision; every file is checked against its manifest checksum. Train on `decision-v7` (ten public datasets plus generated policy and rule records), select on its development split, fit temperature on its calibration split.
- **Evaluation**: `decision-v7` test (trained sources), `transfer-v4` dev and test (sources never trained on), `transfer-v9` test (adds MMLU-Pro, buried evidence, unknowable records). Test splits are read once per released model.
- **Reference point**: [Kev-0.6B](https://huggingface.co/jaredpalmer/kev-0.6b), a LoRA adapter on the same base and data: 0.808 trained-source and 0.642 new-source accuracy on its locked tests.
- **Hardware**: Apple M4, 16 GB unified memory, MLX. A resource guard (`tino/guard.py`) caps MLX memory and pauses or stops training when the Mac runs low, so a run never takes the machine down. Full fine-tuning with AdamW peaked at 10.1 GB, so the backbone uses Adafactor, whose optimizer state is a few MB instead of ~3.5 GB.

## Results

[RESEARCH.md](RESEARCH.md) holds the four research questions, every experiment, and the answers, with paired significance tests and two seeds where it mattered. In short:

- **Recipe (RQ1):** training stability dominates at small data; keep lr ≤ 1e-4 and alpha fixed, and run 2+ seeds. Teacher soft labels are the biggest lever for transfer.
- **Full fine-tune vs adapter (RQ2):** same accuracy; the adapter is 40x fewer trained weights, a 43 MB file, and 4 GB of training memory.
- **Base vs thinking Qwen (RQ3):** thinking adds about 3 points as a zero-shot decider (not significant) at 95x the latency; the student's starting point does not matter.
- **Distillation (RQ4):** an autoregressive Qwen3-4B teacher can be distilled into a 0.6B decision model, without gold labels, reaching 90% of the teacher's out-of-domain accuracy.

All runs, including the ones that failed or were stopped, are in `runs/`.

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
