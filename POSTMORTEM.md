# Postmortem: Building the Two-Stage Recommender

This documents the debugging journey behind the numbers in the README, including the
bugs found, how they were diagnosed, and what actually moved the metrics.
The headline result: the XGBoost reranker started out making recommendations
*worse* than doing nothing (HR@10 0.0875 vs. a 0.1485 baseline from ranking the retrieval scores), and ended at HR@10 0.2910, a 3.3x improvement over where it started and clearly ahead of the stage-1-only baseline.

## Incident 1: Stage 1 and stage 2 were trained on the same data

**Symptom.** Early offline metrics looked strong — suspiciously strong for a
first pass. The underlying cause wasn't visible in any single number; it was
a structural gap in how the data was split.

**Diagnosis.** The original split was a single holdout: interactions were
divided once into `is_train` / `is_test`, and that was it. LightGCN was fit
on the training edges, the semantic user profile was built by averaging the
same training edges' book embeddings, and then the ranker's positive labels
were drawn from *that same training set*. That meant `lightgcn_score` and
`semantic_score` for a ranker "positive" example weren't measuring whether
the model could generalize to a book the user hadn't interacted with yet —
they were partly measuring whether that exact edge had already been trained
on. A model that has memorized an edge will trivially score it well; that's
not a signal that survives contact with genuinely unseen future data, and it
would have made every downstream metric optimistic in a way that wouldn't
hold up. The xgboost ranker was able to "cheat" purely off looking at the
fused score during training and collapsed once faced with the final test set.

**Fix.** Split the training window itself into two disjoint slices instead
of one: a `history` slice (`is_gcn_fit`), which LightGCN and the semantic
profile are fit on, and a separate, later `label` slice
(`is_train & ~is_gcn_fit`), held back from stage 1 entirely and used only as
the ranker's positive labels. Because stage 1 never sees the label window,
`lightgcn_score` / `semantic_score` / `fused_score` for a label-window book
reflect real generalization, not a readout of "was this edge trained on."
This split is threaded through every later stage — `03_make_graph_data.py`
builds the graph from `is_gcn_fit` edges only, and `04_build_ranker_dataset.py`
mines candidates and labels from the history/label split throughout.

**Lesson.** A single train/test split is not automatically leakage-free the
moment you have more than one model in the pipeline. Each *stage* that
learns something from data needs its own accounting of what it's allowed to
have seen — here, stage 1 (LightGCN + semantic profile) and stage 2
(the ranker labels) needed genuinely disjoint windows, not just a shared
"training set" versus "test set" boundary. This is the kind of bug that
doesn't throw an error or even necessarily look wrong in an offline metric —
it just quietly inflates everything until evaluated against truly held-out
future behavior.

## Incident 2: The reranker was worse than not reranking at all

**Symptom.** After the previous fix, Stage 1 (LightGCN + semantic fusion, no reranking) scored NDCG@10 = 0.0702. Stage 1 + XGBoost reranking scored NDCG@10 = 0.0284 — worse than doing nothing, despite the whole point of stage 2 being to improve on stage 1.

**Diagnosis.** The instinct was to blame the model (bad hyperparameters,
wrong objective). Instead, the fix started by inspecting how the *training
data* for the ranker was built. `04_build_ranker_dataset.py` generated
candidates like this:

```python
all_candidate_indices = set(top_k_indices).union(set(label_pos_indices))
```

This force-injects every held-out future positive into the training
candidate pool, regardless of whether it actually scored well enough to
appear in the real top-100. A quick empirical check on the actual dataset:

- Only **~21%** of label positives organically landed in the natural
  top-100 by fused score.
- For **51.5% of users**, *none* of their true positives were in the
  organic top-100 — every one was present purely because of the forced
  union.

So roughly 4 out of 5 "positive" training rows had stage-1 scores that would
never actually surface them into a real top-100 candidate pool. The model
learned that stage-1 scores are an unreliable predictor of relevance, and
leaned on noisier features to explain those positives instead — a lesson
that actively hurt it once deployed, where the candidate pool is strictly
the natural top-100 (a book ranked outside it is a stage-1 miss the reranker
never even sees).

**Fix.** Stop force-injecting out-of-pool positives:

```python
all_candidate_indices = set(top_k_indices)
```

**Impact.** NDCG@10 went from 0.0284 to 0.0709 — the reranker started
roughly matching (rather than badly trailing) the stage-1 baseline.

**Lesson.** A monotonic constraint or clever loss function can't compensate
for a train/serve distribution mismatch. Before tuning a model further,
verify the training distribution actually resembles what the model sees at
inference time. In this case, an empirical check surfaced the bug faster than staring at
hyperparameters would have.

## Incident 3: Monotonic constraints don't guarantee what they sound like they guarantee

Even after fixing Incident 2, XGBoost still trailed the fused baseline
slightly on every metric. Checking feature importances explained why:
`book_age_at_interaction` and `is_long_book` which were both unconstrained features, accounted for **40%** of total importance, more than the
monotonically-constrained stage-1 scores combined. Neither has a strong true
relationship to relevance; the model was very likely fitting spurious
correlations from candidate-generation artifacts.

The underlying misconception: a monotonic constraint only forces the model's
output to be non-decreasing in a *given* feature, holding everything else
fixed. It does not guarantee the model's combined ranking preserves
stage-1's ordering, and it does nothing to stop *unconstrained* features
from dominating the split decisions and injecting noise.

**Fix.** Dropped both unconstrained, weakly-informative features from the
ranker.

**Impact.** XGBoost went from trailing the fused baseline to clearly beating
it on every metric (NDCG@10 0.0702 → 0.0709, HR@5 +34% relative, MRR@10
+37% relative, HR@10 +27%).

**Lesson.** "I added a monotonic constraint" is not the same claim as "this
model can't underperform the baseline." Check feature importances after any
constraint change. A spurious feature can quietly out-vote the ones you deliberately constrained.

## Incident 4: Stage-1 recall was the real ceiling, not the reranker

After the fixes above, XGBoost only marginally beat the fused baseline.
Rather than continuing to tune the ranker, the more useful question was:
how often is the true answer even *retrievable*? Answer: stage-1 recall@100
was only ~20% — in 4 out of 5 cases, the correct next book wasn't in the
candidate pool at all, a ceiling no amount of reranking can fix.

**Fix — three changes made together** (retraining is slow, so they went in
as one batch rather than one at a time):

1. Widened `STAGE_1_TOP_K` from 100 to 200.
2. Added a second, independent retrieval channel: item-item co-occurrence
   ("readers of your history also read...") computed as a sparse book-book
   similarity matrix, unioned with the fused-score candidates.
3. Replaced the semantic user profile's flat mean over all-time history
   with a recency-weighted average, so a recent genre shift isn't diluted
   by years of older reads.

**Impact.** HR@10 0.2004 → 0.2685.

**A useful follow-up diagnostic**: comparing XGBoost against a *naive*
50/50 blend of the fused and co-occurrence scores (no learned model at all)
showed the naive blend alone captured most of the gain (HR@10 0.163 → 0.215). This meant that the co-occurrence channel, not the reranker, was doing most of the
heavy lifting. That reframed where to spend further effort: tuning the
retrieval channel's own hyperparameters, not the ranker.

**Lesson.** When a two-stage system underperforms, check which stage is
actually the bottleneck before optimizing the one that's easiest to iterate
on. A cheap non-learned baseline (the naive blend) is a fast way to
attribute how much of a gain is coming from data/features versus the model.

## Incident 5: Hyperparameter sweep, done cheaply

I confirmed the co-occurrence channel mattered, but its own hyperparameters
(`top_n` neighbors per book, history length used to build the similarity
matrix) were untuned defaults. Sweeping them the expensive way (rebuild
dataset, retrain XGBoost, re-evaluate) would have meant a full pipeline run
per candidate value.

**Approach.** Built a throwaway script that rebuilds just the small,
cheap-to-compute similarity matrix in memory for each candidate
hyperparameter setting, and evaluates it via the same naive-blend proxy from
Incident 4. This skips the expensive dataset rebuild and model retrain
entirely until a good setting was found.

**Result.** Tighter neighbor lists and shorter, more recent history windows
consistently won (`top_n=5, max_hist=50` beat the original `top_n=50,
max_hist=100` on every metric), with returns flattening and reversing on
NDCG/MRR past that point even as HR kept creeping up — the two metrics
disagreed near the boundary, and NDCG/MRR was the tiebreaker.

**Lesson.** Not every hyperparameter needs the full training pipeline to
evaluate. Isolating the cheap, non-learned part of the system and building a
fast proxy metric made a 5-point sweep tractable that would otherwise have
meant hours of retraining per point.

## Incident 6: Out-of-memory on the "successful" dataset rebuild

After locking in the tuned hyperparameters, rebuilding the full ranker
dataset died but only *after* the 32-minute per-user computation loop 
had already finished successfully. The
crash was in final assembly: converting a single Python list of ~50-70
million per-row dicts into a DataFrame, then merging it against book
metadata, on a 16GB machine.

**Fix.** Rewrote the accumulation to flush to disk in chunks (columnar
buffers written as parquet part-files every 15,000 users) instead of holding
everything in memory at once, and replaced the final metadata merge with
precomputed array lookups computed inline during the loop (removing an
expensive hash join over tens of millions of rows entirely).

**Impact.** No OOM, and the run was also slightly faster than before the list-of-dicts-to-DataFrame conversion and the final
merge were themselves slow, not just memory-heavy.

**Lesson.** When a job's final step is "now
assemble everything," that step's own memory/time cost should be considered
part of the job's cost, not an afterthought.

## Smaller issues along the way

- **Book age reference year mismatch**: an eval script used a hardcoded
  `INFERENCE_YEAR` (2024) while the dataset's actual cutoff was 2017 —
  caught and fixed, though it turned out not to be the dominant issue (the
  candidate-pool bug in Incident 2 was much larger).

## Where the numbers ended up

| Stage | HR@5 | HR@10 | NDCG@10 | MRR@10 |
|---|---|---|---|---|
| Broken reranker (start) | — | 0.0875 | 0.0284 | 0.0311 |
| Fused-only baseline | 0.1152 | 0.1625 | 0.0667 | 0.0778 |
| Final reranker | 0.2167 | 0.2910 | 0.1244 | 0.1485 |

## What I'd do differently next time

- Check the training/serving candidate-distribution match *before* writing
  any monotonic constraints or tuning hyperparameters — Incident 2 would
  have been caught in minutes with the same empirical check done later.
- Work on optimizing Stage 1 retrieval scores before moving on to Stage 2 ranking,
would have saved lots of time re-running the entire pipeline
- Build the cheap naive-baseline diagnostic (Incident 4) earlier in the
  process, as a standing tool, rather than after several rounds of "did that
  help?" guessing.
- Treat the "assemble and save" step of any large offline job as seriously
  as the compute-heavy part, memory-profile-wise, before running it for 30+
  minutes.
