# tino research plan

**Goal.** Learn how to turn a small autoregressive decoder into a Jev-style decision model (state + typed questions in, calibrated option probabilities out, one forward pass) on a 16 GB Apple M4, and which way of doing it works best.

## Answers at a glance

| Question | Answer |
|---|---|
| **RQ1.** Which fine-tuning recipe is best? | Stability first: lr ≤ 1e-4, alpha fixed (not scaled with rank), 2+ seeds, and reject runs whose dev log-loss sits at the uniform value; at 4k, 5 of 15 LoRA runs failed: lr 2e-4 on both seeds, rank 64 with alpha scaled to rank, and 2 of the 12 sane configurations by chance. Then supervision: teacher soft labels give calibrated probabilities; their out-of-domain accuracy gain did not hold up over two seeds. Rank above 16 and Brier loss do not help. Need well over 1,000 questions. |
| **RQ2.** Full fine-tune or adapter? | Adapter. Same in-domain accuracy (0.814 vs 0.815, 2 seeds), a lean of about 2 points out of domain within seed noise, and 40x fewer trained weights, a 43 MB file, 4 GB of training memory. Full fine-tuning is the safer pick only with very little data. |
| **RQ3.** Base or thinking Qwen? | As a zero-shot decider, thinking adds 3 points (not significant, p = 0.41) at 95x the latency; it helps on rule-following questions. As the student's starting point, post-trained vs base makes no difference after training. |
| **RQ4.** Can an autoregressive model be distilled into a decision model? | Yes. A half-teacher, half-gold student matches gold-label accuracy and comes out calibrated without temperature scaling, with better out-of-domain probabilities. Teacher only, with no gold labels, also works but costs 5.6 points in domain (significant) for a +1.7-point out-of-domain lean that is not significant over two seeds. |

Seed-to-seed variation out of domain is up to about 4 points, larger than most method differences here; single-seed gaps smaller than that are treated as unproven throughout.

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

Results on development splits (4,000-question subset, 2 epochs). Each cell is trained-source / new-source accuracy. "Collapsed" means the decision head stopped separating options: scores differ by under 0.002, every option gets the same probability, and dev log-loss sits at exactly 1.478, the uniform value.

| ID | Change | Seed 0: trained / new | Seed 1: trained / new | Runs |
|---|---|---|---|---|
| A0 | Baseline: LoRA r16 on all 7 projections, scale 2.0, lr 1e-4, cross-entropy, gold labels | 0.730 / 0.567 | 0.727 / 0.550 | [`20261023-235901`](runs/20261023-235901-abl-adapter-4k), [`20261024-192429`](runs/20261024-192429-abl-adapter-4k-s1) |
| A1 | Full fine-tune instead of LoRA | 0.749 / 0.582 | – | [`20261024-005557`](runs/20261024-005557-abl-finetuned-4k) |
| A2 | LoRA rank 4 | 0.755 / 0.565 | – | [`20261024-023723`](runs/20261024-023723-abl-rank4-4k) |
| A3 | LoRA rank 64, alpha scaled with rank (scale 2.0) | collapsed | – | [`20261024-033412`](runs/20261024-033412-abl-rank64-4k) |
| A3b | LoRA rank 64, alpha fixed at 32 (scale 0.5) | 0.743 / 0.552 | – | [`20261024-182602`](runs/20261024-182602-abl-rank64-a32-4k) |
| A4 | LoRA on attention projections only | stalled at uniform (0.347 / 0.418) | 0.725 / 0.606 | [`20261024-043220`](runs/20261024-043220-abl-attn-only-4k), [`20261024-173603`](runs/20261024-173603-abl-attn-only-4k-s1) |
| A5 | Learning rate 5e-5 | 0.750 / 0.593 | – | [`20261024-052020`](runs/20261024-052020-abl-lr5e-5-4k) |
| A6 | Learning rate 2e-4 | collapsed | collapsed | [`20261024-061741`](runs/20261024-061741-abl-lr2e-4-4k), [`20261024-202356`](runs/20261024-202356-abl-lr2e-4-4k-s1) |
| A7 | Loss: cross-entropy + Brier | 0.682 / 0.552 | – | [`20261024-071515`](runs/20261024-071515-abl-brier-4k) |
| D2a | Distillation, teacher weight 0.25 | 0.755 / 0.573 | – | [`20261024-081309`](runs/20261024-081309-abl-distil-a025-4k) |
| D2b | Distillation, teacher weight 0.5 | collapsed | 0.689 / 0.555 | [`20261024-091047`](runs/20261024-091047-abl-distil-a05-4k), [`20261024-212329`](runs/20261024-212329-abl-distil-a05-4k-s1) |
| D2c | Distillation, teacher only (no gold labels) | 0.723 / 0.611 | – | [`20261024-100808`](runs/20261024-100808-abl-distil-a1-4k) |
| C1 | 1,000 questions | LoRA: 0.348 / 0.394, near uniform | full fine-tune: 0.356 / 0.403, near uniform | [`20261024-014925`](runs/20261024-014925-abl-adapter-1k), [`20261024-021503`](runs/20261024-021503-abl-finetuned-1k) |

**Answer: for a small decision model, the recipe matters in this order.**

1. **Keep training stable; it is the biggest effect at small data.** lr 2e-4 collapsed on both seeds, and rank 64 with alpha scaled to rank collapsed too, because both make each update too large. Rank 64 with fixed alpha trains normally. Sane configurations also failed by chance about 1 time in 6 at 4k (2 of 12: attention-only stalled and teacher-at-0.5 collapsed on one seed, and both trained fine on the other). No LoRA run collapsed at full data (0 of 5). Practical rule: lr ≤ 1e-4, fixed alpha, at least 2 seeds, and reject any run whose dev log-loss sits at the uniform value.
2. **Teacher soft labels change calibration more than accuracy.** At 4k, teacher-only scored 0.611 on new-source dev against 0.559 for gold-only, but at full data with two seeds the out-of-domain gain shrank to +1.7 points (not significant) while the in-domain cost stayed at 5.6 points. Every distilled run came out calibrated (RQ4).
3. **Smaller updates and fewer adapted modules help a little.** lr 5e-5 (0.750 / 0.593) and rank 4 (0.755 / 0.565) both beat the baseline in domain. Attention-only, when it trained, had the best gold-label transfer (0.606). Each is one seed, so these are leads, not settled.
4. **More rank does not help.** Rank 64 (fixed alpha) is no better than rank 16 on new sources.
5. **Plain cross-entropy is the right loss.** Adding a Brier term cost about 5 points in domain and did not improve calibration after temperature scaling.
6. **Data floor.** With 1,000 questions neither method left the uniform-guess plateau in 2 epochs (about 185 steps).
7. **Full fine-tune vs LoRA** at 4k: full fine-tuning was 1.5–2 points ahead and never collapsed. At full data they tie (RQ2).

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
| D1 | `tino-distil`, teacher weight 0.5, full data, 2 seeds | done: [`seed 0`](runs/20261023-153602-tino-distil), [`seed 1`](runs/20261025-014322-tino-distil-s1) |
| D2 | teacher-weight sweep at 4k: 0.25, 0.5, 1.0 | done (RQ1 table, rows D2a–D2c) |
| D3 | `tino-distil-a1`, teacher only (no gold labels), full data, 2 seeds | done: [`seed 0`](runs/20261024-222253-tino-distil-a1), [`seed 1`](runs/20261025-083634-tino-distil-a1-s1) |

The teacher itself, zero-shot and temperature-scaled, scores 0.775 on trained sources and 0.745 on new sources (test splits), at about 330 ms per question.

**Answer: yes. A 0.6B decision model can learn from an autoregressive LLM. What carries over reliably is calibration; extra out-of-domain accuracy does not.**

Two seeds per model, test splits (seed 0 / seed 1):

| Test split | Gold labels (`tino-adapter`) | Half teacher (`tino-distil`) | Teacher only (`tino-distil-a1`) | Teacher |
|---|---|---|---|---|
| decision-v7 (trained sources) | 0.810 / 0.818 | 0.808 / 0.820 | 0.750 / 0.766 | 0.775 |
| transfer-v4 (new sources) | 0.613 / 0.651 | 0.636 / 0.626 | 0.671 / 0.626 | 0.745 |
| transfer-v9 (hardest) | 0.548 / 0.568 | 0.555 / 0.548 | 0.577 / 0.543 | 0.672 |
| Mean of the three, per model | 0.668 | 0.666 | 0.655 | – |
| Fitted temperature | 2.01 / 1.96 | 0.95 / 0.84 | 0.87 / 0.92 | 9.0 |
| Raw ECE, transfer-v4 | 0.152 / 0.111 | 0.049 / 0.040 | 0.097 / 0.040 | 0.245 |
| Brier, transfer-v4 (two-seed mean) | 0.507 | 0.470 | 0.463 | – |

Pooled over both seeds, against the gold-label adapter:

| Teacher only vs gold labels | Accuracy gap | 95% interval | Sign test p | Brier |
|---|---|---|---|---|
| decision-v7 | −5.6 pts | −7.5 to −3.8 | < 0.0001 | 0.345 vs 0.280 (worse) |
| transfer-v4 | +1.7 pts | −0.7 to +4.1 | 0.66 | 0.463 vs 0.507 (better) |
| transfer-v9 | +0.2 pts | −1.9 to +2.1 | 1.00 | 0.544 vs 0.609 (better) |

- **Half teacher, half gold is the best model.** It matches gold-label accuracy everywhere (two-seed means 0.814 / 0.631 / 0.551 vs 0.814 / 0.632 / 0.558; sign tests p ≥ 0.44) and comes out calibrated: fitted temperature 0.84–0.95, so it needs no correction, against about 2.0 for the gold-label adapter. Its raw ECE out of domain is 0.04–0.05 against 0.11–0.15, and its out-of-domain Brier is better on both seeds.
- **Gold labels are not needed.** Trained on the teacher's probabilities alone, the student averages 0.758 in domain and 0.649 out of domain, 87% of the 4B teacher's out-of-domain accuracy in a model a sixth the size, answering in about 50 ms instead of 330.
- **But teacher-only is a trade, not a win.** It loses 5.6 points in domain on both seeds. Its out-of-domain edge is +1.7 points on transfer-v4 and +0.2 on transfer-v9, neither significant. Seed 0's 0.671 was the high end: seed 1 scored 0.626.
- **Calibration is the robust effect.** All four distilled runs needed a temperature of 0.84–0.95; all three gold-label models needed 1.96–2.75.
- **Seed variation dominates accuracy comparisons out of domain.** The same recipe moved by 3.8 points (adapter) and 4.6 points (teacher-only) between seeds.

## Model objectives

Released models live on private Hugging Face repos; each Hub copy was downloaded and re-scored, and reproduces its run's test accuracy exactly. Ablation runs stay as tracked runs only.

| Model | Answers | Recipe | Trained sources | New sources | Calibrated ECE, new | ms per question | Meets bar |
|---|---|---|---|---|---|---|---|
| [`tino-finetuned`](https://huggingface.co/BrandeisPatrick/tino-finetuned) | RQ2 | full fine-tune, Qwen3-0.6B-Base, gold labels | 0.810 | 0.618 | 0.050 | 40 | misses new, ECE |
| [`tino-adapter`](https://huggingface.co/BrandeisPatrick/tino-adapter) | RQ1 baseline, RQ2 | LoRA r16, Qwen3-0.6B-Base, gold labels | 0.810 | 0.613 | 0.055 | 52 | misses new, ECE |
| [`tino-adapter-post`](https://huggingface.co/BrandeisPatrick/tino-adapter-post) | RQ3b | LoRA r16, Qwen3-0.6B (post-trained), gold labels | 0.810 | 0.615 | 0.070 | 51 | misses new, ECE |
| [`tino-distil`](https://huggingface.co/BrandeisPatrick/tino-distil) | RQ4 | LoRA r16, half gold labels, half Qwen3-4B teacher | 0.808 | 0.636 | 0.046 | 51 | misses new |
| [`tino-distil-a1`](https://huggingface.co/BrandeisPatrick/tino-distil-a1) | RQ4 | LoRA r16, Qwen3-4B teacher only, no gold labels | 0.750 | 0.671 | 0.068 | 52 | misses trained, ECE |

Bar set at the start: trained sources ≥ 0.808 and new sources ≥ 0.642 (Kev-0.6B), calibrated ECE ≤ 0.05, training ≤ 8 GB, ≤ 100 ms per question on the M4. Every model trains in 8 GB or less and answers in under 100 ms.

- No released model clears both accuracy bars. The gold-label and half-teacher models meet the in-domain bar; `tino-distil-a1` (seed 0) meets the out-of-domain bar, but its second seed scored 0.766 / 0.626, so that is partly seed luck. A second-seed adapter (not released) scored 0.818 / 0.651 and cleared both, which again shows how much luck sits in these numbers.
- Pick `tino-distil` by default: full in-domain accuracy and calibrated probabilities with no temperature fix. `tino-distil-a1` shows the student can learn without gold labels. `tino-adapter` is the small gold-label baseline.

## Out of scope on this machine

Distilling from the thinking teacher would need reasoning traces for all 15,576 training questions, about 60 hours of generation on the M4. It needs a cloud GPU, so RQ3a measures the thinking model as a decision maker instead.
