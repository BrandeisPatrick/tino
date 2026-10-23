# tino research plan

**Goal.** Learn how to turn a small autoregressive decoder into a Jev-style decision model (state + typed questions in, calibrated option probabilities out, one forward pass) on a 16 GB Apple M4, and which way of doing it works best.

## Shared protocol

Every experiment follows these rules, so results stay comparable.

| | |
|---|---|
| Student base | `Qwen/Qwen3-0.6B-Base` @ `da87bfb6`, unless the experiment changes it |
| Data | [`jaredpalmer/kev-suites`](https://huggingface.co/datasets/jaredpalmer/kev-suites) @ `cc4bac80`; train on `decision-v7/train` (15,576 questions), checksums verified |
| Selection | best `decision-v7/development` accuracy; temperature fitted on `decision-v7/calibration` |
| Test sets | `decision-v7/test` (trained sources), `transfer-v4/test` (new sources), `transfer-v9/test` (hardest). Read once per released model. Ablation runs report development splits only, so test sets never steer a choice. |
| Primary metric | accuracy on new sources (`transfer-v4`), because that is closest to a user's own questions |
| Other metrics | trained-source accuracy, Brier, ECE raw and calibrated, confident-error rate, ms per question, peak memory, training time, artifact size |
| Noise | 95% confidence half-width is about ±3.5 points on `transfer-v4` (764 questions) and ±2.1 on `decision-v7/test` (1,440). Smaller gaps are not claimed without paired tests or extra seeds. |
| Tracking | each run's config, provenance, logs and per-question predictions in `runs/` on GitHub; weights and teacher labels on private Hugging Face repos |
| Machine safety | `tino/guard.py` caps memory and pauses or stops a run before the Mac runs low |

## Research questions

### RQ1. Which fine-tuning recipe is best for a decision model?

An ablation over the knobs that matter, one change at a time from a fixed baseline. To make a sweep affordable, ablations train on a fixed 4,000-question subset for 2 epochs (about an hour each) and are judged on development splits. The winner is then confirmed at full scale.

| ID | Change from baseline (LoRA r16, all projections, lr 1e-4, pointer head, cross-entropy) | Status |
|---|---|---|
| A0 | baseline at 4k questions | queued |
| A1 | method: full fine-tune instead of LoRA | queued (shared with RQ2) |
| A2 | LoRA rank 4 | queued |
| A3 | LoRA rank 64 | queued |
| A4 | LoRA on attention projections only | queued |
| A5 | learning rate 5e-5 | queued |
| A6 | learning rate 2e-4 | queued |
| A7 | loss: cross-entropy + Brier score | queued |
| A8 | method: distillation from the Qwen teacher | queued (shared with RQ4) |

**Answer so far:** none yet beyond RQ2's method comparison.

### RQ2. Full fine-tuning or an adapter?

Same base, data, head, schedule, and numerics; only the backbone training differs.

| Run | Trained sources | New sources | Trainable | Size | Peak memory | Status |
|---|---|---|---|---|---|---|
| `tino-finetuned` seed 0 | 0.810 | 0.618 | 441M | 2.1 GB | 7.7 GB | done |
| `tino-adapter` seed 0 | 0.810 | 0.613 | 10.6M | 43 MB | 4.0 GB | done |
| both, seed 1 | | | | | | queued |
| both at 1k and 4k questions | | | | | | queued |

Still needed: a paired test on the same test questions, a second seed, and a data-size curve. Adapters are often said to win when data is scarce, which the 1k and 4k points test.

**Answer so far:** a tie at full data on one seed. The adapter gets the same accuracy with 40x fewer trained weights, a 43 MB file, and 4 GB of training memory.

### RQ3. Base Qwen or thinking Qwen?

"Thinking" can matter in two places, and both are tested.

| ID | Question | Comparison | Cost | Status |
|---|---|---|---|---|
| 3a | Does reasoning before answering make a better decision maker? | Qwen3-4B-Instruct-2507 (answers directly) vs Qwen3-4B-Thinking-2507 (reasons, then answers), zero-shot on a fixed 200-question development subset (100 per suite) | thinking runs at roughly 10–20 s per question, about 1.5 h | queued |
| 3b | Is a thinking-capable model a better starting point for the student? | `Qwen3-0.6B-Base` vs `Qwen3-0.6B` (post-trained, thinking-capable), identical `tino-adapter` recipe | one 3 h run | queued |

Hypothesis: thinking helps on reasoning-heavy sources (MMLU, policy and deadline rules) but costs 20–50x latency, and the post-trained starting point makes little difference once the decision head is trained.

**Answer so far:** none yet.

### RQ4. Can an autoregressive model be distilled into a decision model?

Teacher: Qwen3-4B-Instruct-2507 (4-bit), read zero-shot. Each question becomes a lettered multiple-choice prompt, and the next-token probabilities of the letters are the soft labels. Student: the `tino-adapter` recipe, with loss α·KL(teacher ‖ student) + (1−α)·cross-entropy on gold. The teacher is softened by a temperature fitted on the calibration split.

Probe on 380 development questions: the teacher scores 0.677 on trained sources and 0.738 on new sources. The students score about 0.79 and 0.60. The teacher knows things the student lacks out of domain, which is what distillation should transfer.

| ID | Run | Status |
|---|---|---|
| D0 | label all splits with the teacher | done: [`tino-teacher-labels`](https://huggingface.co/datasets/BrandeisPatrick/tino-teacher-labels) @ `31e5bfd7`, run [`20261023-140510`](runs/20261023-140510-teacher-qwen3-4b) |
| D1 | `tino-distil`, α = 0.5, full data | done: run [`20261023-153602`](runs/20261023-153602-tino-distil) |
| D2 | α sweep at 4k: 0.25, 0.5, 1.0 (1.0 = teacher only, no gold labels) | queued |

Success means `tino-distil` beats `tino-adapter` on new sources by more than noise, without losing more than a point on trained sources.

**Answer so far: yes, for calibration; not yet for accuracy.**

The teacher, zero-shot and temperature-scaled, scores 0.775 on trained sources and 0.745 on new sources (test splits). Its calibrated ECE is about 0.05.

Paired against `tino-adapter` (same student, gold labels only) on identical test questions:

| Test split | tino-adapter | tino-distil | Accuracy gap | McNemar p | Brier improvement, 95% interval |
|---|---|---|---|---|---|
| decision-v7 (trained sources) | 0.810 | 0.808 | −0.2 pts | 0.88 | +0.005, from −0.014 to +0.026 |
| transfer-v4 (new sources) | 0.613 | 0.636 | +2.4 pts | 0.12 | **+0.063, from +0.037 to +0.091** |
| transfer-v9 (hardest) | 0.548 | 0.555 | +0.6 pts | 0.62 | **+0.076, from +0.055 to +0.098** |

- **Calibration transfers.** The distilled student's probabilities are clearly better out of domain (Brier improves by 0.06–0.08, intervals exclude zero). It needs no temperature correction: its fitted temperature is 0.95, against 2.01 for the adapter and 2.45 for the full fine-tune. Its raw ECE is 0.03–0.05, against 0.10–0.18 for the others.
- **Nothing is lost in domain.** Trained-source accuracy is unchanged.
- **Out-of-domain accuracy barely moves.** The +2.4 points on transfer-v4 test is not significant (p = 0.12), and the development split shows no gain (0.596 vs 0.594). Most of the teacher's 0.745 does not come across. This matches Kev's finding that a 0.6B student is capacity-limited out of domain.
- The α sweep (D2) tests whether more weight on the teacher moves accuracy, and whether gold labels are needed at all (α = 1.0).

## Model objectives

Released models live on private Hugging Face repos. Ablation runs stay as tracked runs only.

| Model | Answers | Recipe | Status |
|---|---|---|---|
| [`tino-finetuned`](https://huggingface.co/BrandeisPatrick/tino-finetuned) | RQ2 | full fine-tune, Qwen3-0.6B-Base | done |
| [`tino-adapter`](https://huggingface.co/BrandeisPatrick/tino-adapter) | RQ1 baseline, RQ2 | LoRA r16, Qwen3-0.6B-Base | done |
| [`tino-distil`](https://huggingface.co/BrandeisPatrick/tino-distil) | RQ4 | LoRA r16 + Qwen3-4B-Instruct teacher | done |
| `tino-adapter-post` | RQ3b | LoRA r16, Qwen3-0.6B (post-trained, thinking-capable) | queued |

Bar for every released model:

- Trained sources ≥ 0.808 (Kev-0.6B). New sources ≥ 0.642 (Kev-0.6B), stretch 0.70.
- Calibrated ECE ≤ 0.05 and confident errors ≤ 3%.
- Trains in ≤ 8 GB peak and answers in ≤ 100 ms per question on the M4.

## Out of scope on this machine

Distilling from the thinking teacher would need reasoning traces for all 15,576 training questions, about 60 hours of generation on the M4. It needs a cloud GPU, so RQ3a measures the thinking model as a decision maker instead.
