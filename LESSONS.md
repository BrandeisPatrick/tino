# What tino taught us

Nine lessons from training a 0.6B Jev-style decision model on a 16 GB Mac, October 21–25, 2026. Each lesson comes with the table behind it. The charts are on the published pages in [`docs/pages/`](docs/pages/), and every number below is read from the run records in [`runs/`](runs/INDEX.md). The research questions themselves are answered in [RESEARCH.md](RESEARCH.md).

**The thesis: how you train a 0.6B decision model matters far less than what it learns from and which random seed you drew.**

## 1. The training method is a wash. Choose on cost.

Full fine-tuning, LoRA adapters, and a post-trained starting checkpoint all land at the same in-domain accuracy. Seven runs across four recipes sit inside a 1.3-point range (0.808–0.821). The only thing that moved a model out of that cluster was changing what it learned from.

| Recipe | Trained sources (test) | New sources (test) | Trainable weights | Published size | Training memory | Fitted temperature |
|---|---|---|---|---|---|---|
| Full fine-tune (2 seeds) | 0.815 | 0.613 | 441M | 2.1 GB | 6.5–7.7 GB | 2.45, 2.75 |
| Adapter, gold labels (2 seeds) | 0.814 | 0.632 | 10.6M | 43 MB | 4.0 GB | 2.01, 1.96 |
| Adapter, post-trained start (1 seed) | 0.810 | 0.615 | 10.6M | 43 MB | 4.0 GB | 1.96 |
| Adapter, half teacher (2 seeds) | 0.814 | 0.631 | 10.6M | 43 MB | 4.0 GB | 0.95, 0.84 |
| Adapter, teacher only (2 seeds) | 0.758 | 0.649 | 10.6M | 43 MB | 4.0 GB | 0.87, 0.92 |

- The adapter trains 10.6M weights instead of 441M, ships as 43 MB instead of 2.1 GB, and peaks at 4 GB of memory.
- The full fine-tune is more overconfident: it needs a temperature of 2.45–2.75 to fix its probabilities, against about 2.0 for the adapter.
- With 4,000 training questions the full fine-tune led by 1.5–2 points, so it is the safer pick only when data is scarce (lesson 6).

## 2. Seed luck is larger than any method difference.

Train the same recipe twice with different seeds and new-source accuracy moves by up to 4.6 points. No gap between recipe means is that large. A single-seed result under about 4 points is noise until a second seed says otherwise.

| Recipe | Seed 0 | Seed 1 | Span |
|---|---|---|---|
| Full fine-tune | 0.618 | 0.607 | 1.0 pts |
| Adapter, gold labels | 0.613 | 0.651 | 3.8 pts |
| Adapter, half teacher | 0.636 | 0.626 | 1.0 pts |
| Adapter, teacher only | 0.671 | 0.626 | 4.6 pts |

New-source test accuracy (transfer-v4). Largest gap between two recipe means: 3.6 points (teacher only vs full fine-tune). The within-recipe spread is wider than the between-recipe spread.

- The first teacher-only run scored 0.671 and looked like the best model of the project. Its second seed scored 0.626.
- Paired tests on identical questions measure test-set sampling. They say nothing about training randomness, which turned out to be the bigger source of noise.
- Two seeds of the baseline told us more than eight single-seed hyperparameter variants.

## 3. Out of domain, model size is the ceiling, not the recipe.

The 4B teacher answers new-source questions at 0.745 zero-shot. Every 0.6B student, whatever it was trained on, stops between 0.61 and 0.67. The knowledge exists in the teacher and does not fit in the student.

| Model | Parameters | New-source accuracy | Source |
|---|---|---|---|
| tino, all 9 full-data runs | 0.6B | 0.607–0.671 | this repo, transfer-v4 test |
| Kev-0.6B | 0.6B | 0.642 | Kev model card, transfer-v4 test |
| Kev-0.8B | 0.8B | 0.697 | Kev model card |
| Qwen3-4B-Instruct-2507, zero-shot | 4B | 0.745 | this repo, run `20261023-140510` |
| Kev-4B | 4B | 0.838 | Kev model card |
| Kev-9B | 9B | 0.852 | Kev model card |
| Kev-27B | 27B | 0.889 | Kev model card |
| Jev (TypeSafe) | undisclosed | 0.857 | Kev's measurement, development split |

Kev's models use the same data and evaluation suite but their own training code and newer Qwen bases, so the comparison is close, not exact. The trend is what matters: nothing we did at 0.6B moved the cluster, and the Kev family climbs steadily with size. Kev's authors reached the same conclusion for their 0.6B model: capacity, not data, is the bottleneck out of domain. If accuracy on questions unlike your training data matters, the fix is a bigger student.

## 4. Distillation transfers calibration, not knowledge.

Train the student on the teacher's probabilities and its own probabilities come out trustworthy with no correction. Its accuracy stays where the student's size allows. For a model whose whole job is returning probabilities you can act on, that is the more valuable half.

| Property | Transferred? | Evidence, two seeds each |
|---|---|---|
| Calibration | yes | fitted temperature 0.84–0.95 vs 1.96–2.75 for gold-label runs; raw out-of-domain ECE 0.04–0.05 vs 0.11–0.15 |
| Probability quality out of domain (Brier) | yes | better on both seeds, 0.466–0.474 vs 0.485–0.529 |
| In-domain accuracy | kept | half teacher 0.814 vs gold 0.814; teacher only drops to 0.758 |
| Out-of-domain accuracy | no | half teacher 0.631 vs gold 0.632; teacher only +1.7 pts, p = 0.66 |
| The teacher's own overconfidence | no | the teacher needed T = 9.0; the student learned from its temperature-scaled probabilities |

Temperature needed to calibrate each run (1.0 means the raw probabilities are already calibrated): full fine-tune 2.45, 2.75; gold-label adapter 2.01, 1.96; post-trained adapter 1.96; half teacher 0.95, 0.84; teacher only 0.87, 0.92.

Teacher: Qwen3-4B-Instruct-2507 at 4 bits, read zero-shot over lettered options. 14,576 of 15,576 training questions got teacher labels; the 77-option banking questions kept gold labels.

## 5. The teacher weight is a dial between in-domain and out-of-domain skill.

More weight on the teacher buys a little transfer and costs in-domain accuracy. Half and half gets the calibration for free and gives up nothing. That is the recipe to ship. Loss = α · KL(teacher ‖ student) + (1 − α) · cross-entropy on gold labels.

| Teacher weight α | Trained sources (test, 2-seed mean) | New sources (test, 2-seed mean) | Fitted temperature |
|---|---|---|---|
| 0 (gold labels only) | 0.814 | 0.632 | 2.0 |
| 0.5 (half teacher, half gold) | 0.814 | 0.631 | 0.90 |
| 1.0 (teacher only, no gold labels) | 0.758 | 0.649 | 0.90 |

- The in-domain drop at α = 1 is significant (−5.6 points, p < 0.0001). The out-of-domain rise is not (+1.7 points, p = 0.66).
- Gold labels are not required: the teacher-only student still reaches 87% of the teacher's out-of-domain accuracy in a model a sixth the size, answering in about 50 ms instead of 330.
- At 4,000 questions the same dial showed more strongly: teacher only 0.611 on new-source dev against 0.559 for gold only. Full data narrowed it.

## 6. With little data, LoRA is fragile, and the failure has a fingerprint.

At 4,000 questions, 5 of 15 LoRA runs failed. Three failed because of their configuration and two by chance. A failed run is easy to spot: its decision head stops separating options (scores within 0.002 of each other) and dev log-loss freezes at exactly 1.478, the log-loss of guessing every option equally.

| Configuration (4,000 questions, 2 epochs) | Seed 0 | Seed 1 |
|---|---|---|
| Baseline: LoRA r16, lr 1e-4 | trained, 0.567 | trained, 0.550 |
| LoRA rank 4 | trained, 0.565 | – |
| LoRA rank 64, alpha scaled with rank | collapsed | – |
| LoRA rank 64, alpha fixed at 32 | trained, 0.552 | – |
| Attention-only LoRA | stalled | trained, 0.606 |
| Learning rate 5e-5 | trained, 0.593 | – |
| Learning rate 2e-4 | collapsed | collapsed |
| Cross-entropy + Brier | trained, 0.552 | – |
| Teacher weight 0.25 | trained, 0.573 | – |
| Teacher weight 0.5 | collapsed | trained, 0.555 |
| Teacher only | trained, 0.611 | – |

Numbers are new-source development accuracy. Collapsed: option scores within 0.002 of each other. Stalled: never left the uniform plateau.

- Learning rate 2e-4 collapsed on both seeds. Rank 64 with alpha scaled to rank collapsed; with alpha fixed at 32 it trained. Both make each update too large.
- Of the twelve sane configurations, two failed by chance, about one in six. No LoRA run failed at full data (0 of 7), and full fine-tuning did not fail in any of its three runs at 4,000 or more questions.
- Practical rule: learning rate at or below 1e-4, fixed alpha, at least two seeds, and reject any run whose dev log-loss equals the uniform value.
- With 1,000 questions neither method left the plateau in two epochs, about 185 steps. That is below the floor for this recipe.

## 7. Thinking is the wrong tool for a decision model.

Letting Qwen reason before answering adds three points that do not reach significance, at 95 times the latency. A decision model's value is answering in milliseconds, and thinking gives that up.

| Same 200 development questions | Qwen3-4B-Instruct, answers directly | Qwen3-4B-Thinking, reasons first |
|---|---|---|
| Accuracy, all 200 | 0.750 | 0.780 |
| Trained sources (100) / new sources (100) | 0.740 / 0.760 | 0.770 / 0.790 |
| Gap and 95% interval | – | +3.0 pts, −3.0 to +9.0 (p = 0.41) |
| Seconds per question on the M4 | 0.33 | 31.6 |
| Reasoning tokens | 0 | 1,137 on average; 43 of 200 hit the 2,048 cap |
| Where it helped | – | held-out deadline rules 2 of 6 → 6 of 6; SciQ 0.88 → 1.00 |
| Where it hurt | – | QNLI 0.91 → 0.73; MMLU 0.76 → 0.71 |

- Thinking pays off on rule-following records, where working through the rule matters, and loses on questions that reward a quick read.
- As a starting checkpoint for the student, the post-trained, thinking-capable Qwen3-0.6B ended up identical to the base model: 0.810 vs 0.810 in domain, 0.615 vs 0.613 out of domain. It only learned faster in the first epoch.

## 8. Score questions are everyone's weak spot.

Every model picks the right option about 85% of the time and the right yes/no answer about 87% of the time, then scores around 50% when asked for the exact level on a 3-to-5-point scale. No recipe touched that gap.

| Model (seed 0, trained-source test) | Choice | Noul (yes/no) | Score (exact level) |
|---|---|---|---|
| Full fine-tune | 0.876 | 0.863 | 0.508 |
| Adapter, gold labels | 0.845 | 0.871 | 0.583 |
| Adapter, post-trained start | 0.851 | 0.875 | 0.558 |
| Adapter, half teacher | 0.858 | 0.869 | 0.538 |
| Adapter, teacher only | 0.783 | 0.831 | 0.483 |

- Score questions are sentiment and star-rating scales (SST-5, Yelp, Amazon) where the levels are ordered, yet the model treats them as unrelated options.
- The teacher is weak here too (its own Score accuracy on development data was 0.47–0.51), so distillation cannot help.
- An ordinal loss that rewards being one level off is the obvious next experiment. We did not run it.

## 9. On a 16 GB laptop, memory bugs decide feasibility, not hyperparameters.

Three fixes made the whole project possible, and none of them would appear in a results table. The full log with commits is in [docs/ENGINEERING.md](docs/ENGINEERING.md).

| Problem | Symptom | Fix | Effect |
|---|---|---|---|
| AdamW optimizer state for 441M weights | 10.1 GB peak, Mac at 19–24% free memory, guard pausing every step | Adafactor for the backbone | 10.1 → 7.5 GB |
| Loss read before gradients were computed | 9.2 GB peak for a 10.6M-parameter adapter | Evaluate loss and gradients together so gradient checkpointing holds | 9.2 → 4.0 GB |
| Casting the fp32 model to bf16 for publishing | 18% of a smoke model's decisions flipped | Evaluate and publish in the training precision | 0 flipped |
| Two GPU jobs at once | Swap grew 3.6 GB; the guard stopped a 2.6-hour labeling run | Every job runs alone, in one queue, at low priority | no further aborts |
| Queue waited on its own shell | Idle for 5 minutes after labeling finished | Wait on a process ID, not a command-line match | 5 min lost |

## What to do with these results

| If you | Do this | Because |
|---|---|---|
| want probabilities you can act on | LoRA r16 with half-teacher distillation (`tino-distil`) | calibrated out of the box, no accuracy cost (lessons 4, 5) |
| need accuracy on questions unlike your training data | use a bigger student; stop tuning the recipe | teacher 0.745 vs every 0.6B student at or below 0.67 (lesson 3) |
| have fewer than about 5,000 training questions | full fine-tune, lr ≤ 1e-4, two seeds, reject runs at uniform log-loss | LoRA failed 5 of 15 at 4,000; full fine-tuning 0 of 3 (lesson 6) |
| are comparing two recipes | run two seeds before believing a gap under 4 points | seed swing reached 4.6 points (lesson 2) |
| need answers in milliseconds | answer directly; skip thinking models | +3 points, not significant, 95x slower (lesson 7) |
| must rate on an ordered scale | expect about 50% exact-level accuracy; try an ordinal loss | all models 0.48–0.58 on Score vs about 0.85 on Choice (lesson 8) |

| Bar set at the start | Outcome |
|---|---|
| Trained sources ≥ 0.808 (Kev-0.6B) | met by 4 of 5 released models (0.808–0.810); teacher only sits at 0.750 |
| New sources ≥ 0.642 (Kev-0.6B) | met once, by teacher-only seed 0 at 0.671; its second seed scored 0.626. A second adapter seed, not released, scored 0.651 |
| Calibrated ECE ≤ 0.05 | met by `tino-distil` (0.046) and `tino-finetuned` (0.050) |
| Train in ≤ 8 GB; answer in ≤ 100 ms | met by every model |

No single released model clears both accuracy bars, and the bar itself sits inside the seed noise. Knowing that is the result.

## What is not settled

- Most ablations have one seed; the reruns show single runs can swing several points out of domain.
- The thinking comparison used 200 questions and a 2,048-token reasoning cap.
- Distilling from the thinking model would need reasoning traces for all 15,576 training questions, about 60 hours on the M4; it needs a cloud GPU.
- Worth testing next: attention-only adapters with several seeds, lr 5e-5 at full data, an ordinal loss for Score questions, and a larger teacher such as Qwen3.5-9B for labels.
