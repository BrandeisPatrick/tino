# Phase 2: making the four answers hold up

Phase 1 (October 22–25) answered RQ1–RQ4 with 27 training runs at one student size, one teacher, two seeds where it mattered, and Kev's own test suites. This plan lists what a reviewer would still ask for, states the hypotheses before the runs, and prices every block in hours on the 16 GB M4 so the scope can be chosen deliberately. One job runs at a time, under the memory guard, with every run tracked as before.

## What Phase 1 cannot yet claim

| Gap | Why it matters | Fix |
|---|---|---|
| One or two seeds per cell | Out-of-domain accuracy moves up to 4.6 points between seeds; most ablation cells are one seed | Three seeds per full-data recipe, two per ablation cell, means ± SD and pooled paired tests |
| One student size, one teacher | Nothing says whether the ordering of recipes, or calibration transfer, survives a bigger student or teacher | Students at 0.6B and 1.7B (4B optional); teachers at 1.7B, 4B and 8B |
| Test suites chosen by the reference model's author | transfer-v4 and transfer-v9 are Kev's suites; a method tuned on Kev's development splits may look better on them than elsewhere | Five suites from the same dataset revision that none of our models or ablations ever saw: SemIf, Scienthoon, the binding diagnostic, the public-pool trio (ARC, OpenBookQA, CommonsenseQA) and transfer-r3 |
| Reference numbers from other code | Kev-0.6B's 0.808 / 0.642 come from Kev's harness and splits | Load Kev-0.6B's published adapter and head into this harness and score it on the same splits |
| Thinking compared on 200 questions | ±6-point interval cannot separate +3 from zero | 800 development questions (±3.4) |
| Many paired tests, no correction | Fourteen comparisons in Figure 3 of the article; one of them at p = 0.02 | Holm correction within each question's family; a claim needs the corrected interval to exclude zero and the sign to agree across seeds |
| Precision tied to fp32 | Larger students need a bf16 backbone; a size comparison across precisions is confounded | A 0.6B adapter trained with the same bf16 backbone as the 1.7B runs |

## Rules fixed before any run

- Primary metric stays transfer-v4 test accuracy. Secondary: decision-v7 test, transfer-v9 test, Brier, raw and calibrated ECE, the fitted temperature, and the five new suites.
- Selection on development splits only. Test splits and the new suites are read once per released model, after selection. Ablations report development splits.
- Every full-data recipe gets three seeds; every retained ablation cell two. Seeds change data order, LoRA initialisation and dropout.
- Comparisons are paired on the same questions, pooled over seeds: percentile bootstrap intervals and the exact McNemar test, Holm-corrected within each research question. A difference is claimed when the corrected interval excludes zero and every seed agrees in sign; otherwise it is reported as a lean.
- Nothing is tuned on the new suites. All released and new models are scored on them from their published weights, in one pass, at the end.
- Hypotheses H1–H6 below are recorded now; the write-up reports each as confirmed, refuted or undecided.

## Hypotheses

- **H1 (RQ1, stability).** With lr ≤ 1e-4 and fixed alpha, no full-data run collapses (0 of ≥ 15), and at 4,000 questions sane configurations collapse at most 1 time in 6 over the added seeds. Attention-only adapters do not beat all-projection adapters on transfer-v4 once each has three seeds.
- **H2 (RQ2, method).** At 0.6B the adapter and the full fine-tune stay within ±2 points on trained sources and the adapter's out-of-domain lean stays inside [−1, +4] points after Holm correction: a tie. At 1.7B the adapter gains at least 5 points on transfer-v4 over the 0.6B adapter (Kev's ladder: 0.620 at 0.6B, 0.790 at 4B, 0.796 at 8B on its development split).
- **H3 (RQ3, thinking).** On 800 development questions, reasoning first stays within ±4 points of direct answering overall, gains on rule-following sources, and loses on QNLI; the post-trained start stays indistinguishable from the base start with a second seed.
- **H4 (RQ4, calibration transfer).** Every distilled student, from every teacher size, needs a temperature in [0.8, 1.1]; students trained on an *unsoftened* teacher (temperature 1 instead of the fitted 9.0) come out over-confident (temperature ≥ 1.5), which would show that the fitted softening, not the teacher per se, carries the calibration.
- **H5 (RQ4, teacher size).** Teacher-only students improve on transfer-v4 monotonically with teacher size (1.7B < 4B < 8B) and the in-domain cost of teacher-only training shrinks with the 8B teacher but does not vanish; half-teacher students match gold-label accuracy for every teacher.
- **H6 (new tests).** The ordering of recipes on the public-pool trio matches transfer-v4; distilled students have lower raw ECE than gold-label students on every new suite; on SemIf perturbation pairs, no recipe keeps both siblings correct more than 60% of the time.

## Experiment matrix

Hours are wall-clock on the M4 from measured Phase 1 rates (0.6B adapter or distil run: 3.2 h; full fine-tune: 3.1 h; 4,000-question ablation: 0.9 h; 4B teacher labelling: 2.6 h; thinking model: 31.6 s per question). Rates for 1.7B come from the smoke run recorded below.

### Tier 1: seeds, statistics, new tests (about 50 h)

| ID | Runs | Hours | Answers |
|---|---|---|---|
| S1 | Third seed of tino-finetuned, tino-adapter, tino-distil, tino-distil-a1 | 4 × 3.2 = 13 | RQ2, RQ4: three-seed means and corrected tests |
| S2 | Second seed of tino-adapter-post | 3.2 | RQ3b with two seeds |
| S3 | Teacher weight 0.25 and 0.75 at full data, two seeds each | 4 × 3.2 = 13 | RQ4: the full α curve |
| S4 | Half-teacher student with the teacher unsoftened (T = 1), two seeds | 2 × 3.2 = 6.5 | H4: what carries calibration |
| S5 | Second seed for the eight informative 4k cells (full fine-tune, rank 4, rank 64 fixed alpha, lr 5e-5, Brier, α 0.25, α 1.0, 1k full fine-tune) | 8 × 0.9 = 7 | RQ1: every cell at two seeds |
| T1 | Thinking vs direct on 600 more development questions | 5.5 | RQ3a at ±3.4 points |
| N1 | Score every released and new model on the five new suites, plus two probes: option-order permutation consistency and a state-removed control | < 1 | H6; how much of each answer comes from the state |
| B1 | Kev-0.6B in this harness (convert its adapter and head; verify against its card: 0.801 development, 0.620 transfer-v4 development) | engineering, < 0.5 h compute | A same-code reference row in every table |

### Tier 2: teacher size (about 28 h)

| ID | Runs | Hours | Answers |
|---|---|---|---|
| K1 | Label all splits with Qwen3-1.7B (4-bit, non-thinking) and Qwen3-8B (4-bit) | 1.5 + 6 | Teacher accuracy ladder, soft labels at three sizes |
| K2 | Half-teacher students from the 1.7B and the 8B teacher, two seeds each | 4 × 3.2 = 13 | H5 |
| K3 | Teacher-only students from the 8B teacher, two seeds | 2 × 3.2 = 6.5 | H5: does a better teacher close the in-domain gap |

### Tier 3: student size (about 43 h without M3 and M4)

| ID | Runs | Hours | Answers |
|---|---|---|---|
| M0 | 0.6B adapter with the bf16 backbone, one seed | 3.2 | The precision control for every 1.7B comparison |
| M1 | 1.7B adapter, three seeds | 3 × 8 = 24 | H2: the size step for the adapter |
| M2 | 1.7B half-teacher student (4B teacher), two seeds | 2 × 8 = 16 | Does calibration transfer survive a bigger student |
| M3 | 1.7B full fine-tune in bf16 with Adafactor, if it fits under the guard, one or two seeds | engineering + about 12 per seed | H2 at the second size |
| M4 | 4B adapter on a locally quantized 4-bit base, 4,000 questions, one seed (optional) | about 6 | A third point on the size curve, ablation scale only |

### Out of scope on this machine

Distilling from the thinking teacher at full data (about 60 h of generation), 8B students, and more than one seed at 4B.

## Order of work

1. Tier 1 in one queue (S1–S5, T1), about two days, while B1 and N1 are built and run between jobs.
2. Tier 2 labelling and students, about one day.
3. Tier 3 once M0 and the smoke run confirm memory and time; M3 and M4 only if the hours fit.
4. Re-score everything on the new suites, update `RESEARCH.md`, regenerate figures, add a scaling figure (student size × teacher size) and the new-suite table to the article as v0.2.

## How it runs

Everything above except M3 and M4 is queued, in this order, in `queue/pending.txt` by `scripts/queue_run.sh` (one job at a time, low priority, under the guard): the Tier 1 jobs in `scripts/phase2_tier1.jobs`, then `scripts/phase2_tier23.jobs`. The first suite evaluation (`configs/suite-eval.json`: the Phase 1 models and the Kev port) runs right after the first Tier 1 job; the last job scores every Phase 2 model on the same suites (`configs/suite-eval-phase2.json`). Lines can be added to the pending file while the queue runs. The Kev port is `scripts/kev_port.py`; the suite evaluation and probes are `scripts/eval_suites.py`.

## 1.7B smoke run, measured October 29

Adapter recipe on `Qwen/Qwen3-1.7B-Base` (revision `ea980cb0`), bf16 backbone, 240 questions, 25 steps, under the guard: run [`20261029-170321`](../runs/20261029-170321-tino-adapter-1.7b-smoke). 18.5M trainable parameters; 4.9 GB peak MLX memory against 4.0 GB for the 0.6B adapter; 206 tokens per second against 470 for the 0.6B at the same step, so 2.3 times slower. Training completed. The guard then stopped the run during the final evaluation because system swap had grown 3.1 GB: the Mac already had 6 GB swapped out before the run started, and evaluation batches use a 4,096-token budget. Two consequences for Tier 3: evaluation at 1.7B uses the training budget of 2,048 tokens, and memory-heavy applications are closed for the long runs. Estimate: about 8 h per full-data 1.7B run (7.4 h of training plus slower evaluations).

## Total

Tier 1 about 50 h, Tier 2 about 28 h, Tier 3 about 43 h without M3 and M4: about five days of continuous compute, one job at a time, with the laptop usable but slower throughout. Tier 1 alone is about two days; Tiers 1 and 2 about three.

## Progress

### October 29, evening

- **Queue.** Started 20:59 with the Tier 1 jobs. The first, `tino-adapter-s2`, was stopped by the guard at step 680 of 2,881 when system swap grew past the 3 GB rule while the machine was busy with other work (run [`20261029-205913`](../runs/20261029-205913-tino-adapter-s2); [engineering log](ENGINEERING.md)). The rule is now 6 GB in every config, and the job restarted at 23:07 as run `20261029-230744`. 34 jobs follow it.
- **B1 and the Kev port, done.** Run [`20261029-214858-suite-eval`](../runs/20261029-214858-suite-eval) scored the nine Phase 1 models and Kev-0.6B on the never-seen suites and the two probes (79 minutes, 0.04 s per question). The port is validated: on the two development splits Kev's card reports, it scores 0.809 on decision-v4 (card 0.801) and 0.631 on transfer-v4 (card 0.620), within a point of both, with a fitted temperature of 1.74.

Accuracy on suites no run was selected on (raw, one pass each):

| Model | SemIf (252) | Scienthoon (873) | Binding (560) | Public trio (348) | transfer-r3 (1,260) |
|---|---|---|---|---|---|
| tino-finetuned | 0.623 | 0.450 | 0.562 | 0.500 | 0.642 |
| tino-finetuned-s1 | 0.615 | 0.424 | 0.573 | 0.514 | 0.620 |
| tino-adapter | 0.591 | 0.564 | 0.639 | 0.532 | 0.644 |
| tino-adapter-s1 | 0.587 | 0.477 | 0.636 | 0.540 | 0.675 |
| tino-adapter-post | 0.667 | 0.507 | 0.677 | 0.489 | 0.643 |
| tino-distil | 0.639 | 0.494 | 0.677 | 0.543 | 0.653 |
| tino-distil-s1 | 0.619 | 0.481 | 0.662 | 0.540 | 0.664 |
| tino-distil-a1 | 0.690 | 0.515 | 0.648 | 0.543 | 0.658 |
| tino-distil-a1-s1 | 0.635 | 0.532 | 0.646 | 0.489 | 0.642 |
| kev-0.6b | 0.599 | 0.392 | 0.652 | 0.537 | 0.636 |

First reading, before any test (two seeds per recipe, so differences under about 4 points are noise, as in Phase 1):

- The full fine-tune is the weakest recipe on the binding diagnostic (0.56–0.57 against 0.64–0.68 for every adapter-based model) and on transfer-r3 (0.62–0.64 against 0.64–0.68): the same direction as its Phase 1 out-of-domain gap.
- The distilled students sit at or above the gold-label models on every suite except Scienthoon, and their raw calibration is better on every suite (ECE 0.04–0.20 against 0.11–0.35 for the gold-label models), as H6 predicts.
- Scienthoon splits by question type: the queue choice is answered at 0.71–0.87 by the tino models and 0.53 by Kev; the priority score sits at 0.16–0.31 for every model; the anger question at 0.40–0.56.
- Kev-0.6B lands with the tino adapters on binding, the public trio and transfer-r3, and below them on SemIf and Scienthoon.
- SemIf's own protocol (family-balanced accuracy, the `semif` entry of each model): 0.60–0.72 on the 144 authored rows, where SemIf reports 0.813 for Qwen3.5-4B's direct logits, and 0.54–0.71 on the 108 perturbed rows (reference 0.723). On the 36 perturbation bases, the share where all three perturbations (option reversal, criterion wrapper, irrelevant context) are answered right is 0.39–0.61; `tino-adapter-post` reaches 0.61, so H6's 60% line is not settled on 36 bases. The base is right but a perturbation wrong on 3–17% of bases.
- Option-order probe: reversing the options leaves the answer unchanged on 0.59–0.97 of choice questions, by model and suite. `tino-distil` (0.84 on transfer-v4, 0.91 on SemIf, 0.88 on binding) and Kev-0.6B (0.85, 0.83, 0.91) are the steadiest; the teacher-only students the least (0.68–0.80 on the same suites); the public trio is the least stable for everyone (0.63–0.78).
- State-removal probe: with the state blanked, accuracy drops by 15–27 points on transfer-v4 development, 5–19 on binding and 27–40 on SemIf. Every model reads the state.
- One defect surfaced and was fixed: the SemIf perturbation groups came out empty because perturbed rows point at their base through `_meta.parent_id`, not the group id. `scripts/eval_suites.py` now follows `parent_id` and reports SemIf's family-balanced accuracy; `--rescore` recomputed this run from its saved predictions, so no model was re-run.
