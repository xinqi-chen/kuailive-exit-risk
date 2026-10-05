# Reproduce from raw data

Use Python 3.13. The repaired run was checked in a fresh Python 3.13.5 environment using the pinned dependencies:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
make test
```

Run from the repository root. The repository is intended to run from source; keep `configs/` and `scripts/` with `src/`.

## Inputs

```bash
make download
```

The downloader fetches the exact files in `configs/raw_sha256.json` from KuaiLive-M3 revision `417923d442a793b3e92cc7e079dd4b73fe706c30`, verifies every SHA-256, and stops on any mismatch. The formal build repeats this verification. The full raw inputs occupy roughly 40 GB; allow additional space for features, context, arrays and models. Download and training require substantially more time and memory than smoke. No raw or row-level files are redistributed.

You may reuse already downloaded files by setting `KUAILIVE_DATA_DIR` to their directory. Smoke needs only `user_id_set.csv`, `live_interaction.csv`, and `live_room_meta.parquet`; formal runs additionally need the like/comment CSVs and all 18 embedding shards. `KUAILIVE_PROCESSED_DIR` overrides the processed root. These paths need not point to a development checkout.

## Independent smoke

```bash
make smoke
```

This scans the raw interactions, selects 20 deterministic users, constructs visits, labels and features, trains M1_browse, and evaluates H30. It creates a fresh ignored `raw-smoke-<timestamp>/` directory and never requires a formal matrix or saved model. Co-viewer counts cover those 20 users. It uses no event logs or embeddings. Its small metrics verify execution, not formal performance.

The repaired locked-environment check produced 34,723 visits, 86,499 training and 22,309 validation rows; primary H30 had 2,358 points, log loss 0.272030, Brier 0.074498, AUC 0.700303 and AP 0.210314. A clean source export without `.git`, historical matrices, or saved models was also checked. Platform warnings about denied CPU cache inspection do not prevent Arrow execution.

An optional `--formal-feature-root /path/to/model_features` compares semantic row keys and shared non-room fields after common float32 conversion. Room fields are excluded because smoke has a different audience population. An absent optional reference does not affect standalone execution.

## Full formal pipeline

```bash
# Use a new, unused run ID so the immutable run directory is not overwritten.
export KUAILIVE_RUN_ID="formal-repro-$(date +%Y%m%d-%H%M%S)"
make features
make formal
make learning-curve
make audit
make report
```

`features` independently selects A/B/C, normalizes raw records, constructs R0–R4 visits, verifies the entry vocabulary against A/train, chooses cutpoints using A/train, builds personal history, reads room/event/content context from raw files and writes checksummed shards. It refuses to overwrite an existing run directory. To rerun, use a new run ID. Aggregate outputs default to `results/$KUAILIVE_RUN_ID/`; `KUAILIVE_RESULTS_DIR` overrides that location.

`formal` fits fixed ablations, legacy selected M_all parameters and shuffled-block controls on A, evaluates B, then freezes executable source/configuration, model bytes, input shards and training provenance before evaluating C. The two phases can also be run separately with `scripts/run_final_ablation.py --phase train-b` and `--phase eval-c`. Editing any frozen input or executable file invalidates C evaluation; use a fresh run to incorporate changes.

The learning curve uses nested SHA-256 prefixes of A, the same validation users and fixed parameters. Its 1,000-user endpoints reuse the formal models. The audit reports label sensitivity only. `report` verifies the aggregate metric hashes before generating `docs/results.md`. Results are exploratory; the user bootstrap holds the fitted models fixed.

On the Apple-silicon Mac used for the reported run, `features` took about 1.3 h and `formal` about 2.8 h with a peak of about 8 GB RAM; the learning curve and audit take minutes. `report` rewrites `docs/results.md` from the new run's aggregate files.

The data card (`docs/data_card/`) is regenerated separately with `pip install duckdb`, then `python docs/data_card/compute_stats.py && python docs/data_card/build_data_card.py`. The README figure is regenerated with `python docs/figures/make_figures.py`.

For CPU/platform differences, compare metrics with a stated numerical tolerance rather than expecting identical model bytes. Dependency versions, seed, feature grid, file hashes and per-model iterations are captured in the manifests. Upstream embedding generation and serving latency are outside this repository's reproducibility scope.
