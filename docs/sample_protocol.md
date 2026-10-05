# Sample construction and label protocol

## Data and populations

The analysis uses KuaiLive-M3 revision `417923d442a793b3e92cc7e079dd4b73fe706c30` over
`[2026-03-03 16:00 UTC, 2026-03-31 16:00 UTC)`. Training, validation, and test boundaries are
March 21, March 26, and March 31 at 16:00 UTC. User selection is deterministic: SHA-256 of the
stringified user ID in ascending order, with the frozen A/B/C exclusions and test-window activity
rule in `configs/cohorts.yaml`.

A has 1,000 training/validation users, B has 300 development-test users, and C has 1,000 held-out test users. Evaluable users and rows depend on the repaired risk set and are reported in the current results. Configs pin user-selection hashes; manifests record actual selection and row counts. No row-level records or IDs are distributed.

## R0–R4 visit construction

1. **R0 — normalize and screen.** Parse timestamps with column-specific units, stringify IDs,
   remove exact duplicate source rows, apply session quality checks, and detect cross-room overlaps.
   Overlap detection compares each record with the maximum end time of all earlier records for that
   user, rather than only the immediately preceding row.
2. **R1 — stitch midnight splits.** Join a segment ending at local midnight (`23:59:59.999`) to
   the matching next-day segment beginning at midnight for the same user and room. Keep source-row
   lineage. Unmatched midnight endings are flagged for review; endings at the observation boundary are censored rather than called exits.
3. **R2 — merge same-room fragments.** Combine same-user, same-room segments separated by at most
   `g = 1 s` into one visit. This removes record fragmentation without bridging longer returns.
4. **R3 — handle overlapping rooms locally.** Retain the visit, but exclude prediction windows whose
   future outcome overlaps another room visit. This avoids dropping an entire long session because
   one short period is ambiguous.
5. **R4 — classify the end.** Let `T_b` be the recorded broadcast end and `e` the visit end. If
   `e < T_b − 5 s`, the visit ended by a voluntary user exit. If `T_b − 5 s ≤ e ≤ T_b + 60 s`, it
   ended because the broadcast ended (a competing event, with event time `T_b`). If
   `e > T_b + 60 s`, the record is marked as an anomaly and produces no labels. A visit beginning
   after `T_b` is excluded. If `T_b` is missing, the broadcast was still live when the data end (the
   room table records end times only inside the observation window), so a visit that ends before the
   data end is a voluntary exit.

The selected estimand is **E2-V**, voluntary exit; broadcast endings censor it. **E2-A** counts a
broadcast end as an event. All models in this repository are trained and evaluated on E2-V; E2-A
appears only in the protocol audit (label counts under different tolerance settings).

## Risk sets and labels

Prediction times are the starts of 12 non-uniform intervals, using cutpoints
`[0, 5, 10, 15, 20, 25, 35, 50, 75, 120, 225, 345, 600]` seconds. Cutpoints are chosen using
training users only. A row is at risk only while the visit and broadcast are both live. E2-V labels
are assigned as follows:

| Condition at interval end | Result |
|---|---|
| Voluntary end in the interval, with sufficient observation buffer | Positive |
| Voluntary exit after the interval, or the visit ends with the broadcast and the interval closes before `T_b − 5 s` | Negative |
| The visit ends with the broadcast and the interval reaches `T_b − 5 s` | Censored |
| Overlap, low-quality visit, record anomaly, or incomplete observation | Not evaluable |

Every interval outcome uses a 5 s observation buffer. H30 is evaluated on `t = 30, 60, …, 600 s`;
the main summary uses landmarks through 570 s, with 600 s retained as a sensitivity row. The H30
outcome is voluntary exit in `(t, t + 30 s]` while the room remains live.

## Time-window features

All event windows use `[t − w, t)`: the lower bound is included and events at `t` are excluded. Completed personal visits enter history only after `end + max(g, δ−)` and strictly before the query. This delay makes same-room stitching and the early broadcast-end band observable. Visits with ambiguous cross-room overlaps are excluded from completed history. Historical survival uses exact millisecond event times and Kaplan–Meier risk sets, rather than integer-second equality.

Gift/like/comment history flags are aggregated over original source rows through preserved lineage, including every midnight continuation. Released aggregate actions are assumed available after this history delay; this repository cannot verify real serving ingestion latency.

Room context unions overlapping or contiguous raw fragments, without bridging positive gaps using a later return. Counts exclude the querying user and use all released sampled users in target rooms, independently of A/B/C membership. The coverage flag may include the current record's start at `t`; it indicates observable sampled coverage, not complete platform coverage. Other starts/ends and actions are strictly pre-query. Timestamp endpoints are authoritative for reconstructed visit quality; raw reported-duration discrepancies are retained as R0 diagnostics rather than treated as true viewing duration.

## Audit and scientific scope

Current reconstructed counts, end-delay histograms, overlapping-room diagnostics and nine `δ−/δ+` settings are generated under `results/formal-v1.1.0/exp_protocol_audit/`. Sensitivity changes labels and risk sets; it is not evidence that either target can be predicted better. Earlier audit counts are archived and should not be used to describe the repaired run.

These counts describe the released sample, not the platform population. Voluntary exit is an inferred logged event, not independently verified intent. Broadcast-end tolerance, informative censoring and upstream content availability remain assumptions. See [`results.md`](results.md) for model performance and uncertainty.
