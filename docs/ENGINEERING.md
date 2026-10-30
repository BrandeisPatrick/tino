# Engineering log

What broke while running this project on a 16 GB Apple M4, what it looked like, what fixed it, and the commit. Newest last.

| Date | Problem | Symptom | Fix | Commit or run |
|---|---|---|---|---|
| Oct 22 | No limit on what a training job could take from the machine | First smoke run reached 10.1 GB and left the Mac at 19–24% free memory on every step | `tino/guard.py`: MLX memory cap, pause under 25% free, abort under 12% free or on 3 GB of swap growth; batches cut to 2,048 padded tokens | `46cfd5d` |
| Oct 22 | AdamW keeps two fp32 moments for all 441M trainable weights | 10.1 GB peak for the full fine-tune | Adafactor for the backbone (factored second moments, no first moment); AdamW kept for the small head | `bcd1d81` |
| Oct 22 | Casting the fp32-trained model to bf16 for publishing | On a smoke model, 18% of decisions changed and max logit difference was 4.2 | Evaluate and publish in the precision the model was trained in; `publish_dtype` set to null | `1b0d7fa` |
| Oct 23 | Lazy dtype conversion ran inside the first training step | One-off 9 GB spike at step 1 of the first adapter smoke run, with both copies of the weights alive | Materialize the conversion before training and reset the peak-memory counter | `5bc9bd9` |
| Oct 23 | `loss.item()` was read before the gradients were evaluated | 9.2 GB peak for a 10.6M-parameter LoRA run: the forward pass ran alone and kept every activation for the pending backward pass, defeating gradient checkpointing | Evaluate loss, accuracy and gradients in one `mx.eval` call | `079ce83`; peak 9.2 → 4.0 GB, results identical |
| Oct 23 | A quick distillation check ran while the teacher was labeling | Swap grew 3.6 GB; the guard aborted the 2.6-hour labeling run with 6 of 7 splits uploaded | Labeling resumes from the Hub; every job since runs alone in one queue at `nice -n 10` | run `20261023-131057`, resumed in `20261023-140510` |
| Oct 23 | A resumed labeling run skipped uploaded splits but did not record them | The resumed run could not write the dataset manifest | Record skipped splits in the summary | `66e5d71` |
| Oct 23 | The queue's wait loop used `pgrep -f` on the labeler's command line | It matched its own parent shell and never exited; the queue idled for 5 minutes | Wait on a process id instead | `c41ecc3` |
| Oct 24 | Four of the first nine 4,000-question LoRA runs produced identical dev log-loss at every evaluation | Collapsed runs score every option within 0.002 of each other; dev log-loss sits at 1.478, the uniform value; the temperature fit hits its 0.1 floor | Found by inspecting `evals.jsonl`; seed reruns separated configuration failures (lr 2e-4, rank 64 with scaled alpha) from chance ones (attention-only, teacher weight 0.5); `collect_results.py` now flags collapsed runs | `84c39a7`, `05791c4` |
| Oct 25 | A headline result rested on one seed | Teacher-only distillation scored 0.671 on new sources; the second seed scored 0.626 | Two seeds for every claim that matters; the write-up treats single-seed gaps under 4 points as unproven | `99096d0`, `449ef6a` |
| Oct 29 | The guard's swap-growth rule fired on a healthy run | `tino-adapter-s2` stopped at step 680 of 2,881 with 4.0 GB of MLX memory and 57% free system memory; swap had grown past 3 GB because of other processes, including a CPU-side evaluation dry run started alongside it | Swap-growth limit raised from 3 to 6 GB in every config; the free-memory pause (25%) and abort (12%) thresholds, which are what protect the machine, unchanged; no CPU-heavy work beside a running job; the job re-queued | run `20261029-205913`, re-run follows the suite evaluation |
| Oct 29 | The SemIf perturbation metric found no groups | `groups: 0` for all ten models in the first suite evaluation | Perturbed SemIf rows point at their base row through `_meta.parent_id`, not the base's `group_id`; `scripts/eval_suites.py` now follows `parent_id`, reports SemIf's family-balanced accuracy, and has `--rescore` to recompute a finished run from its saved predictions, so nothing was re-run | run `20261029-214858`, rescored |
| Oct 29 | Another session loaded an 8B model in Ollama (7.4 GB, wired) while the queue ran | `tino-adapter-s2` aborted twice (swap growth, then free memory at 9% while paused) and the next job started under the same pressure | Pending jobs held aside; the model unloaded with `ollama stop`; the sessions on the machine asked by message to stay off models over 2 GB while the queue runs; `scripts/queue_run.sh` now waits for 35% free memory and no Ollama model before starting each job | runs `20261029-205913`, `20261029-230744` |

## Things that worked and are worth keeping

- Refusing to train from uncommitted code (`tino/track.py`), so every run maps to a commit.
- Pushing the run directory at every evaluation, so a dead run still has a record.
- Checking every data file's SHA-256 against the suite manifest on load (`tino/data.py`).
- Re-downloading each published model from the Hub and re-scoring it before calling it done. All five reproduced their test accuracy exactly.
- Saving per-question predictions, which made paired McNemar tests and bootstrap intervals possible after the fact (`scripts/report.py`).
- `--smoke` before every long run. The three stopped smoke runs cost minutes; the bugs they caught would have cost hours.
