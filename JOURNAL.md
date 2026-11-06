# Journal

What we are doing, why, and what comes next, one entry per working day. **Now** is kept current, and new entries go at the end. Earlier days (October 21–29) are told in [docs/TIMELINE.md](docs/TIMELINE.md). Per-run facts are in [runs/INDEX.md](runs/INDEX.md). Results and their significance tests are in [RESEARCH.md](RESEARCH.md) and [docs/PHASE2.md](docs/PHASE2.md).

## Now (November 5)

- **Nothing is running.** The Phase 2 queue and the generative arm finished on November 5 at 03:46. Every job exited cleanly, the queue runner has stopped, and `queue/pending.txt` is empty. The queue's own log is archived in [docs/queue/](docs/queue/).
- **Next, when Patrick asks** (on November 5 he said "not now"):
  1. The final Phase 2 write-up. It resolves H1–H6 in docs/PHASE2.md, using the never-seen-suite evaluation of 23 models ([run](runs/20261104-064327-suite-eval-phase2)) for H6. It also updates LESSONS.md, the figures (`scripts/figures.py`), the article (`scripts/build_article.py`) and the published pages.
  2. Optional, for the parameter-efficiency question: a 4B point at the 4,000-question scale, and a third 1.7B generative seed.
  3. Optional engineering: make `scripts/think_eval.py` resumable. An abort lost 3.6 hours on October 31.
- **Weights.** Only the five released models are on the Hugging Face Hub (see [docs/MODELS.md](docs/MODELS.md)). The other 58 runs' weights (15.3 GB) sit in `checkpoints/` on the Mac only; whether to upload them is still Patrick's decision. Every run's per-question predictions are in git, so every reported number and the final write-up can be rebuilt without them. Re-scoring a model on new data, or using it, needs its weights.
- **Teacher labels** for the 1.7B, 4B and 8B teachers are on the Hub (`BrandeisPatrick/tino-teacher-labels`, revisions in each teacher run's `final.json`).

## October 30

- Tier 1 continued: third seeds of the full-data recipes, students from the unsoftened teacher, and the teacher-weight curve.
- At 13:18 a chat in the OpenAgent app loaded an 8B model through Ollama during `tino-distil-raw`. Free memory fell to 13%, and the guard paused the run for 271 seconds; no steps were lost ([engineering log](docs/ENGINEERING.md)). OpenAgent's model was switched to qwen3:1.7b. Later Patrick shut OpenAgent down ("local testing is done"), and Ollama was stopped.
- Patrick asked whether we could run more or go faster. Only one MLX job fits at a time on 16 GB. He chose to keep the full plan rather than trim it.
- Patrick raised a new question: are decision models more parameter-efficient than generative ones, because they need not spend capacity on producing language? It was recorded in RESEARCH.md and not scheduled ("we deal with this later").

## October 31

- The second seeds of the teacher-weight curve and of the post-trained start finished.
- Patrick: "let's finish what's missing." The missing piece was a fine-tuned generative counterpart of the decision model at the same sizes. That is the generative arm: [`tino/generative.py`](tino/generative.py) and [`scripts/train_gen.py`](scripts/train_gen.py). It uses the same input rows, batches and adapter recipe as the decision model, and the model writes its answer. Accuracy is read from the written answer, with option scores used for calibration.
- A mistake: the CPU checks for the new code ran beside the thinking evaluation. Swap grew 8.3 GB, and the guard aborted that evaluation at question 475 of 800, losing about 3.6 hours. The rule since then: heavy checks run only with `queue/hold` set, at a job boundary.
- The first GPU smoke run ran out of memory generating 16 answers at once. Answers are now generated one question at a time, each with its own cache.
- Patrick: "full speed, plus analysis and reflections." The generative arm moved ahead of the remaining ablation seeds. The interim three-seed analysis went into docs/PHASE2.md, with paired tests in [`scripts/seed_compare.py`](scripts/seed_compare.py).

## November 1

- Thinking before answering, against answering directly, on 800 development questions: 0.786 against 0.749, +3.8 points (interval +1.0 to +6.5). Thinking took 34 seconds per question against 0.31.
- Both generative smoke runs passed. The two 0.6B generative runs scored 0.791, 0.621 and 0.555 (in domain, new sources, transfer-v9), against 0.811, 0.637 and 0.557 for the decision adapter.
- The second seeds of the 4,000-question ablations ran. Teacher labels were made with Qwen3-1.7B and Qwen3-8B, which scored 0.560 and 0.736 on new sources.

## November 2

- Tier 2: students trained from the 1.7B and 8B teachers.
- The disk was down to 3.4 GB free. Patrick: "remember to cleanup for unused resource actively." The duplicate `best` copy of every finished run was removed (16.3 GB), along with test leftovers. With his OK, five cached 4-bit teacher models were removed too (16.1 GB). The standing routine now: after each finished run, regenerate the index, and remove `best` once its weights are byte-identical to `publish`.
- The bf16 control (a 0.6B adapter with the 1.7B runs' precision) started.

## November 3

- 1.7B adapter, three seeds: 0.841, 0.671 and 0.594, against 0.811, 0.637 and 0.557 at 0.6B. All three gains are claims. The 5-point gain predicted on new sources (H2) did not appear.
- The first 1.7B distillation seed.

## November 4

- The second 1.7B distillation seed. Distillation leaves accuracy unchanged against the 1.7B adapter. Its raw calibration error is lower (0.040 against 0.061), but the difference disappears once a temperature is fitted, the same pattern as at 0.6B.
- The never-seen-suite evaluation of all 23 Phase 2 models ran from 06:43 to 10:24. Its analysis is part of the final write-up.
- The two 1.7B generative runs started, at 10:26 and 19:00.

## November 5

- The queue finished at 03:46. Paired tests were added for size, for distillation at 1.7B, and for decision against generative ([`docs/seed_compare.json`](docs/seed_compare.json)). The results are in RESEARCH.md, at the end of the parameter-efficiency section. A 0.6B decision model is as accurate as the 1.7B generative one, and it answers about nine times faster. The decision model's lead did not shrink with size, so the idea that writing language costs capacity is not confirmed. One 1.7B generative seed diverged during training.
- Patrick deferred the final write-up ("not now").
- Patrick asked whether everything is recorded, so that this chat could be closed and the local copy deleted. Git was fully pushed. The gaps were:
  - the day-by-day record after October 29;
  - a stale schedule in docs/PHASE2.md;
  - working rules that existed only in the assistant's memory for this folder;
  - the queue logs;
  - 15.3 GB of weights that exist only on the Mac.
- The first four gaps are closed by this journal (Patrick: "we should have a journal.md record what we are doing"), by [`CLAUDE.md`](CLAUDE.md) with the working rules, and by [docs/queue/](docs/queue/).
