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

**Interim results** (4,000-question subset, 2 epochs, development splits only, one seed each; seed reruns are queued):

| ID | Change | Trained-source dev | New-source dev | Calibrated ECE (new) | Run |
|---|---|---|---|---|---|
| A0 | baseline: LoRA r16, all projections, lr 1e-4, cross-entropy | 0.730 | 0.567 | 0.060 | [`20261023-235901`](runs/20261023-235901-abl-adapter-4k) |
| A1 | full fine-tune instead of LoRA | 0.749 | 0.582 | 0.073 | [`20261024-005557`](runs/20261024-005557-abl-finetuned-4k) |
| A2 | LoRA rank 4 | 0.755 | 0.565 | 0.133 | [`20261024-023723`](runs/20261024-023723-abl-rank4-4k) |
| A3 | LoRA rank 64 (alpha 128, scale 2.0) | collapsed (0.336) | collapsed (0.343) | – | [`20261024-033412`](runs/20261024-033412-abl-rank64-4k) |
| A4 | LoRA on attention projections only | collapsed (0.347) | collapsed (0.418) | – | [`20261024-043220`](runs/20261024-043220-abl-attn-only-4k) |
| A5 | learning rate 5e-5 | 0.750 | 0.593 | 0.073 | [`20261024-052020`](runs/20261024-052020-abl-lr5e-5-4k) |
| A6 | learning rate 2e-4 | collapsed (0.339) | collapsed (0.449) | – | [`20261024-061741`](runs/20261024-061741-abl-lr2e-4-4k) |
| A7 | loss: cross-entropy + Brier | 0.682 | 0.552 | 0.052 | [`20261024-071515`](runs/20261024-071515-abl-brier-4k) |
| D2a | distillation, teacher weight 0.25 | 0.755 | 0.573 | 0.095 | [`20261024-081309`](runs/20261024-081309-abl-distil-a025-4k) |
| D2b | distillation, teacher weight 0.5 | collapsed (0.341) | collapsed (0.394) | – | [`20261024-091047`](runs/20261024-091047-abl-distil-a05-4k) |
| D2c | distillation, teacher only (weight 1.0, no gold labels) | 0.723 | 0.611 | 0.056 | [`20261024-100808`](runs/20261024-100808-abl-distil-a1-4k) |
| C1a | LoRA baseline at 1k questions | collapsed (0.348) | collapsed (0.394) | – | [`20261024-014925`](runs/20261024-014925-abl-adapter-1k) |
| C1b | full fine-tune at 1k questions | 0.356 | 0.403 | 0.044 | [`20261024-021503`](runs/20261024-021503-abl-finetuned-1k) |

"Collapsed" means the decision head stopped separating options: its scores differ by less than 0.002, so every option gets the same probability (dev log-loss exactly 1.478, the uniform value).

What the interim results show:

- **Training stability is the first-order effect at small data.** 4 of 9 LoRA runs at 4k collapsed (rank 64, attention-only, lr 2e-4, distillation at weight 0.5), while 0 of 3 LoRA runs at full data did. Distillation at 0.5 trained fine at full data, so collapse is partly chance. Seed reruns of the collapsed settings will measure how much.
- **Smaller updates are safer.** Rank 4 and lr 5e-5 both beat the baseline (0.755 and 0.750 vs 0.730 on trained sources). Rank 64 with scale-with-rank alpha and lr 2e-4 both collapsed. A rank-64 rerun with fixed alpha (scale 0.5) is queued as the fair test of rank.
- **More teacher, better transfer.** On new sources, teacher-only distillation scores 0.611, the best of any 4k run and above both full-data gold-label models (0.594 and 0.598). Gold-only scores 0.567.
- **Brier loss does not help.** It costs about 5 points on trained sources and does not improve calibration after temperature scaling.
- **1,000 questions is too few for this recipe.** Both methods stay near the uniform-guess plateau (about 185 steps).
- A full-data, teacher-only model (`tino-distil-a1`) is queued to confirm the transfer result on the test splits.

### RQ2. Full fine-tuning or an adapter?

Same base, data, head, schedule, and numerics; only the backbone training differs. Two seeds each at full data, test splits:

| Test split | Full fine-tune: seed 0 / seed 1 / mean | Adapter: seed 0 / seed 1 / mean |
|---|---|---|
| decision-v7 (trained sources) | 0.810 / 0.821 / **0.815** | 0.810 / 0.818 / **0.814** |
| transfer-v4 (new sources) | 0.618 / 0.607 / **0.613** | 0.613 / 0.651 / **0.632** |
| transfer-v9 (hardest) | 0.544 / 0.533 / **0.539** | 0.548 / 0.568 / **0.558** |

| Cost | Full fine-tune | Adapter |
|---|---|---|
| Trainable parameters | 441M | 10.6M |
| Published size | 2.1 GB | 43 MB |
| Peak training memory | 7.7 GB (before the loss/grad fix; lower after) | 4.0 GB |
| Fitted temperature (overconfidence) | 2.45, 2.75 | 2.01, 1.96 |

Runs: [`finetuned s0`](runs/20261022-225150-tino-finetuned), [`finetuned s1`](runs/20261024-142648-tino-finetuned-s1), [`adapter s0`](runs/20261023-085627-tino-adapter), [`adapter s1`](runs/20261024-110527-tino-adapter-s1).

At smaller data (4,000 questions, development splits) the full fine-tune led: 0.749 vs 0.730 in domain, 0.582 vs 0.567 on new sources. At 1,000 questions neither method trained.

**Answer: the adapter, on cost; the two tie on accuracy in domain, and the adapter leans ahead out of domain.**

- In domain they are identical (0.815 vs 0.814, two seeds each).
- Out of domain the adapter averages about 2 points higher on both new-source test sets. Its own two seeds differ by 3.8 points, though, so this is a consistent lean, not a proven win.
- The full fine-tune is more overconfident (needs a temperature of 2.5–2.75 vs 2.0) and costs 40x the trainable weights, 50x the file size, and about twice the memory.
- With only 4,000 questions the full fine-tune was ahead by 1.5–2 points, and LoRA runs at that size often collapsed (see RQ1). So with little data, full fine-tuning is the safer choice.
- Seed variation out of domain (up to about 4 points) is larger than any method difference measured here.

### RQ3. Base Qwen or thinking Qwen?

"Thinking" can matter in two places, and both are tested.

| ID | Question | Comparison | Cost | Status |
|---|---|---|---|---|
| 3a | Does reasoning before answering make a better decision maker? | Qwen3-4B-Instruct-2507 (answers directly) vs Qwen3-4B-Thinking-2507 (reasons, then answers), zero-shot on a fixed 200-question development subset (100 per suite) | thinking runs at roughly 10–20 s per question, about 1.5 h | done: run [`20261023-185358`](runs/20261023-185358-think-eval) |
| 3b | Is a thinking-capable model a better starting point for the student? | `Qwen3-0.6B-Base` vs `Qwen3-0.6B` (post-trained, thinking-capable), identical `tino-adapter` recipe | one 3 h run | done: run [`20261023-204028`](runs/20261023-204028-tino-adapter-post) |

Hypothesis: thinking helps on reasoning-heavy sources (MMLU, policy and deadline rules) but costs 20–50x latency, and the post-trained starting point makes little difference once the decision head is trained.

**Answer so far (3a): no significant gain, at about 95x the cost.**

Same 200 development questions (100 trained-source, 100 new-source), both models 4-bit on the M4:

| Model | Trained sources | New sources | Calibration error | Seconds per question |
|---|---|---|---|---|
| Qwen3-4B-Instruct-2507, answers directly | 0.740 | 0.760 | 0.23–0.26 | 0.33 |
| Qwen3-4B-Thinking-2507, reasons first (mean 1,137 tokens) | 0.770 | 0.790 | 0.19–0.23 | 31.6 |

- Over all 200 questions thinking is +3.0 points (0.780 vs 0.750), but the 95% interval is −3.0 to +9.0 and McNemar p = 0.41 (thinking alone right on 21, instruct alone on 15).
- Gains concentrate on rule-following records, as hypothesized: held-out deadline rules go from 2 of 6 to 6 of 6, return-window and quantity-limit records also improve. Thinking does worse on QNLI (0.91 to 0.73) and slightly worse on MMLU (0.76 to 0.71).
- 43 of 200 reasoning traces (21.5%) hit the 2,048-token cap before finishing, so thinking may be slightly understated. A higher cap costs proportionally more time.
- For a decision model that must answer in milliseconds, direct answering is the practical choice. Thinking is worth it only for rule-heavy decisions where 30 seconds per question is acceptable.

**Answer (3b): no difference once trained; the post-trained start only learns faster.**

Identical `tino-adapter` recipe from `Qwen3-0.6B-Base` vs the post-trained, thinking-capable `Qwen3-0.6B`, paired on the same test questions:

| Test split | From base | From post-trained | Gap | McNemar p |
|---|---|---|---|---|
| decision-v7 (trained sources) | 0.810 | 0.810 | 0.0 pts | 1.00 |
| transfer-v4 (new sources) | 0.613 | 0.615 | +0.3 pts | 0.93 |
| transfer-v9 (hardest) | 0.548 | 0.532 | −1.7 pts | 0.18 |

- Brier scores are also indistinguishable (all intervals include zero), and both need a similar temperature (1.96 vs 2.01).
- The post-trained start led on development accuracy through the first epoch (0.715 vs 0.693 at step 360, 0.798 vs 0.768 at step 1,440), then the base caught up.
- Post-training for chat and thinking does not carry into the decision head. Either start works; the base model keeps the setup simplest.


### RQ4. Can an autoregressive model be distilled into a decision model?

Teacher: Qwen3-4B-Instruct-2507 (4-bit), read zero-shot. Each question becomes a lettered multiple-choice prompt, and the next-token probabilities of the letters are the soft labels. Student: the `tino-adapter` recipe, with loss α·KL(teacher ‖ student) + (1−α)·cross-entropy on gold. The teacher is softened by a temperature fitted on the calibration split.

Probe on 380 development questions: the teacher scores 0.677 on trained sources and 0.738 on new sources. The students score about 0.79 and 0.60. The teacher knows things the student lacks out of domain, which is what distillation should transfer.

| ID | Run | Status |
|---|---|---|
| D0 | label all splits with the teacher | done: [`tino-teacher-labels`](https://huggingface.co/datasets/BrandeisPatrick/tino-teacher-labels) @ `31e5bfd7`, run [`20261023-140510`](runs/20261023-140510-teacher-qwen3-4b) |
| D1 | `tino-distil`, α = 0.5, full data | done: run [`20261023-153602`](runs/20261023-153602-tino-distil) |
| D2 | α sweep at 4k: 0.25, 0.5, 1.0 (1.0 = teacher only, no gold labels) | queued |

Success means `tino-distil` beats `tino-adapter` on new sources by more than noise, without losing more than a point on trained sources.

**Answer so far: yes for calibration (robust); accuracy and Brier gains are within seed variation.**

The teacher, zero-shot and temperature-scaled, scores 0.775 on trained sources and 0.745 on new sources (test splits). Its calibrated ECE is about 0.05.

Paired against `tino-adapter` (same student, gold labels only) on identical test questions:

| Test split | tino-adapter | tino-distil | Accuracy gap | McNemar p | Brier improvement, 95% interval |
|---|---|---|---|---|---|
| decision-v7 (trained sources) | 0.810 | 0.808 | −0.2 pts | 0.88 | +0.005, from −0.014 to +0.026 |
| transfer-v4 (new sources) | 0.613 | 0.636 | +2.4 pts | 0.12 | **+0.063, from +0.037 to +0.091** |
| transfer-v9 (hardest) | 0.548 | 0.555 | +0.6 pts | 0.62 | **+0.076, from +0.055 to +0.098** |

- **Calibration transfers.** The distilled student needs no temperature correction: its fitted temperature is 0.95, against 1.96–2.01 for both adapter seeds and 2.45 for the full fine-tune. Its raw ECE out of domain is about 0.05, against 0.11–0.15 for the adapter seeds.
- **Seed caution.** A second adapter seed scored 0.651 on transfer-v4 test (Brier 0.485), 3.8 points above the first (McNemar p = 0.007). Paired tests cover test-set sampling, not training randomness, and out of domain that randomness is about 4 points. Against the better adapter seed, distillation's Brier edge shrinks to 0.02. A second distillation seed is queued; accuracy and Brier claims wait for it.
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
| [`tino-adapter-post`](https://huggingface.co/BrandeisPatrick/tino-adapter-post) | RQ3b | LoRA r16, Qwen3-0.6B (post-trained, thinking-capable) | done |

Bar for every released model:

- Trained sources ≥ 0.808 (Kev-0.6B). New sources ≥ 0.642 (Kev-0.6B), stretch 0.70.
- Calibrated ECE ≤ 0.05 and confident errors ≤ 3%.
- Trains in ≤ 8 GB peak and answers in ≤ 100 ms per question on the M4.

## Out of scope on this machine

Distilling from the thinking teacher would need reasoning traces for all 15,576 training questions, about 60 hours of generation on the M4. It needs a cloud GPU, so RQ3a measures the thinking model as a decision maker instead.
