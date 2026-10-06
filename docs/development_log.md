# Development history and earlier results

These notes record exploratory stages from an earlier code base. They are not results of this repository's code; current results are in [`results.md`](results.md).

The final study (README, [`results.md`](results.md)) was preceded by a series of development stages,
labelled D1–D8 below. Earlier stages used **older sampling and labelling rules**, so their numbers are
not comparable with the final results. They are listed here to show which directions were tried and
how large their effects were. No code for these stages is included in this repository.

## Setup shared by the earlier stages (D1–D4)

- 300 users, all used for training, validation **and** testing; test = the same users' last 5 days
  (no unseen-user test).
- Target: any recorded end of viewing, including broadcast end ("all-cause"); no midnight stitching,
  no merging of same-room fragments, and whole sessions were dropped when their end time conflicted
  with the broadcast end.
- Dynamic predictions were made on a fixed grid `t = 30, 60, …, 600` s after entry.
- Model: LightGBM unless stated; metric: test log loss (lower is better).

Naming in the tables: **M0** current state at prediction time; **M1** = M0 + viewing history;
**M2** = M1 + current content embedding (with availability and age).

## Stages

| Stage | What was tried | Conclusion |
|---|---|---|
| D1 | Prediction at entry: will the visit last ≤10 s, 10–60 s or >60 s (three classes) | Viewing history gives the largest gain; content adds little (Table 1) |
| D2 | Dynamic prediction: end within the next 60 s; nested M0→M1→M2; CatBoost and an MLP as alternative model families | Same picture; other model families no better (Table 2) |
| D3 | Shorter windows (10 s, 30 s); features from the user's historical survival curve | 30 s adopted to match the 30 s grid; survival-curve features ≈ +0.0004 (Table 3) |
| D4 | Historical content profiles of the user and profile–content matching (LightGBM and small neural networks); a user-ID check; likes/comments; comment-content profiles | No reliable dynamic gain; entry-time profile gains are smaller than what a plain user-ID feature gives (Tables 1, 2, 3) |
| D5 | Rebuilt the sample protocol (see [`sample_protocol.md`](sample_protocol.md)); added unseen-user test cohort; switched to non-uniform interval survival models | Found false exits from midnight partitioning, sub-second same-room re-records and delayed end records after a broadcast ends |
| D6 | Fixed the conversion from interval risk to a 30 s probability; main target changed to voluntary exit | Survival model matches the fixed-grid 30 s model |
| D7 | Browsing-rhythm features; training-size curve (75–300 users) | Small, consistent gain; more users help very little |
| D8 | 1,000 training users, all feature blocks, a small hyper-parameter search, evaluation on cohort C | Historical basis; the current release rebuilds inputs and retrains after the audit |

## Table 1 — entry-time three-class prediction (D1, D4)

| Model | Log loss |
|---|---:|
| Constant (class frequencies) | 0.9821 |
| M0 (entry context only) | 0.8478 |
| M1 (+ viewing history) | 0.7783 |
| M2 (+ current content embedding) | 0.7763 |
| M2 + historical-content metadata | 0.7750 |
| + user's historical content profiles | 0.7733 |
| + profile–content matching features | 0.7711 |
| Small neural network, pooled history | 0.7813 |
| Small neural network, learned matching | 0.7819 |
| **Check:** M1 + user ID as a categorical feature | 0.7678 |
| **Check:** M2 + user ID | 0.7651 |

Adding the user ID alone improves M1 by 0.0105, twice the gain of the whole profile/matching chain
(0.0052). The profile gains therefore cannot be distinguished from the model simply recognising users,
which is possible because test users were also training users.

## Table 2 — dynamic prediction, end within the next 60 s (D2, D4)

| Model | Log loss |
|---|---:|
| Constant | 0.5290 |
| M0 | 0.4737 |
| M1 | 0.4540 |
| M1 + content availability/age | 0.4539 |
| M2 | 0.4531 |
| M2 + historical-content metadata | 0.4532 |
| + historical content profiles | 0.4535 |
| + profile–content matching | 0.4533 |
| Small neural network, pooled history | 0.4559 |
| Small neural network, learned matching | 0.4575 |
| CatBoost M1 / M2 | 0.4540 / 0.4531 |
| MLP (RealMLP) M1 / M2 | 0.4537 / 0.4547 |

## Table 3 — dynamic prediction, end within the next 30 s (D3, D4)

| Model | Log loss |
|---|---:|
| M0 | 0.3536 |
| M1 | 0.3411 |
| M2 | 0.3407 |
| M2 + smoothed historical survival curve | 0.3403 |
| M2 + raw historical exit ratios and risk-set sizes | 0.3403 |
| M1 + own likes/comments | 0.3409 |
| M1 + own and others' likes/comments | 0.3409 |
| M2 + own and others' likes/comments | 0.3405 |
| + comment-content profile (comments matched to the following 60 s of content) | 0.3406 |
| + comment-content profile (nearest segment within ±30 s) | 0.3407 |
| + profile split by same/other streamer | 0.3408 |

With a 10 s window, M1 and M2 scored 0.1838 and 0.1837.

**Noise check (D3 setup):** adding 128 randomly shuffled copies of the embedding columns to M1 + content
availability/age *worsened* log loss by about 0.0003, a change that a bootstrap over users still flags as
"significant". Differences below roughly 0.0005 in Tables 2–3 should therefore be read as no effect.

## Post-audit release (v1.1.0)

The public pipeline now rebuilds every feature from the pinned raw dataset revision. Repairs cover fractional-time exit labels and empty-history defaults; window-local overlap exclusion; past-only room, history and event features; complete midnight action lineage; consistent H30 conversion and placebo handling; and the independent raw-to-smoke path. These changes and the model definition changed together, so v1.0.0 and v1.1.0 metrics are not a causal comparison of any individual fix.

M0 now contains elapsed watch time, broadcast age and its missing flag, entry autoplay/follow state, and one-hot `enter_live_action` plus `live_source_category` joined from the visit table by `visit_id`. Category vocabularies are fit only on A/train; categories observed fewer than 1,000 times and missing or unseen values map to `other`. The hashed one-hot column to category mapping is in `configs/entry_category_vocab.json`. All later nested models, placebos and `M_all_tuned` inherit these state features. The frozen interval edges remain `[0, 5, 10, 15, 20, 25, 35, 50, 75, 120, 225, 345, 600]` seconds. Fixed models use `num_leaves=31`, `learning_rate=0.05`, `min_child_samples=100`, at most 3,000 rounds and 100-round early stopping. The previously selected tuned model keeps `num_leaves=127`, `learning_rate=0.03`, `min_child_samples=300`, `feature_fraction=1.0`, with the same 3,000/100 limits; no new grid search was run.

### Pre-training row-count gate

The previous feature matrix contained 6,413,828 rows. The rebuilt matrix contains 8,265,047 rows, exactly the sum of `evaluable` counts in its label-flow manifest. The previous matrix did not contain A/test rows; these add 997,941 interval rows and 467,486 H30 rows and are not used for fitting or reported evaluation. On overlapping splits, the increases are consistent with the repaired risk set and per-window overlap rules:

| Cohort/split/task | Previous rows | Rebuilt rows | Change |
|---|---:|---:|---:|
| A/train interval | 3,351,447 | 3,523,367 | +5.1% |
| A/validation interval | 935,700 | 988,373 | +5.6% |
| A/validation H30 | 429,572 | 465,875 | +8.4% |
| B/test interval | 234,197 | 246,728 | +5.4% |
| B/test H30 | 103,502 | 112,357 | +8.6% |
| C/test interval | 940,423 | 1,002,460 | +6.6% |
| C/test H30 | 418,987 | 460,460 | +9.9% |

The new matrix has 8,265,047 unique `(cohort, query_kind, user_id, visit_id, prediction_time_ms, offset_seconds)` keys, zero missing entry one-hot cells, zero rows whose one-hot sum differs from one in either field, and zero `room_start_missing=1` rows. A/train interval exit rates by frozen bin start at 43.76% for 0–5 seconds and then decline before rising at later intervals. The selected A/B/C user hashes and interval edges match the frozen cohort and protocol configuration. These checks found no unexpected row-count or encoding issue.

### Frozen evaluation and evidence limits

The run freezes cohort hashes, feature order, model parameters, placebo definitions, source/configuration, feature shard hashes and the H30 reporting cutoff (`t ≤ 570 s`) before C evaluation. Models and any choices are assessed on A/validation and B; C contributes no supervised training rows or labels and was also evaluated as a test cohort in earlier model experiments. “Once” refers to the single frozen C evaluation within the post-audit run. Room-context features can include already observed logs from B/C users; their future labels are not used. Paired user bootstrap intervals condition on the fitted models and do not include training randomness, shared-room dependence or multiplicity. Small increments do not establish a prediction ceiling or prove a feature family is unhelpful. Any possible explanation for weak content gains remains a hypothesis; limitations include upstream embedding provenance and serving latency, sampled co-viewers, an exit-intent proxy target and censoring assumptions.

### Current results

On C's 449,700 primary H30 points from 981 users, M0 log loss is 0.327980 and M1 is 0.316145 (paired ΔLL −0.011835, 95% user interval [−0.013130, −0.010535]). This supports a large conditional contribution from viewing history. Beyond history, fixed `M_all` reduces log loss by 0.001935 (0.612%); previously selected tuning reduces it by another 0.000164. The C interval indicates resolved conditional gains for browsing rhythm (−0.000975), co-viewer context (−0.000648) and recent like/comment events (−0.000300); local time (−0.000044), past action rates (+0.000153) and content embeddings (−0.000121) have intervals that include zero. M_all tuning's interval excludes zero by a very small margin. These intervals condition on the trained models and are exploratory, unadjusted for multiple comparisons. Placebo scores are diagnostics rather than noise floors. The entry features are included in M0, so their separate contribution is not identified. Full B/C tables, interval results, learning curve, parameter iterations and bootstrap intervals are in [`results.md`](results.md).

Selected H30 log-loss values from the earlier and rebuilt runs are retained for context:

| Cohort | Model | Earlier n | Earlier log loss | Rebuilt n | Rebuilt log loss |
|---|---|---:|---:|---:|---:|
| B | M0 | 101,192 | 0.359522 | 109,833 | 0.340724 |
| B | M1 | 101,192 | 0.347023 | 109,833 | 0.328641 |
| B | M_all fixed | 101,192 | 0.344516 | 109,833 | 0.326836 |
| B | M_all tuned | 101,192 | 0.344638 | 109,833 | 0.326474 |
| C | M0 | 409,271 | 0.348622 | 449,700 | 0.327980 |
| C | M1 | 409,271 | 0.335612 | 449,700 | 0.316145 |
| C | M_all fixed | 409,271 | 0.333466 | 449,700 | 0.314210 |
| C | M_all tuned | 409,271 | 0.333473 | 449,700 | 0.314046 |

These are not a before/after estimate of a particular code fix: the rebuilt run changes labels, feature availability, event lineage, room/history semantics, and M0 entry state together, and it evaluates a different risk set. The old per-run artifacts are excluded from the public repository.
