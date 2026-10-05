"""Bounded, raw-only formal pipeline. All row-level outputs stay under data/."""
from pathlib import Path
from collections import Counter
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import load_config
from .data.cohorts import select_cohorts, extract_visits, _csv_batches
from .features.groups import MODEL_FEATURES, enrich_points
from .features.history import add_visit_history
from .features.browsing import personal_features, local_time_features, r2_room_visits, room_features_for_room, ROOM
from .features.events import asof_action_counts, parse_event_time, EVENT_FEATURES
from .features.content import join_content_bucket, CONTENT_META, EMBEDDING_COLUMNS
from .labels.periods import choose_cutpoints, make_periods, h30_from_visits
from .provenance import sha256, source_manifest, runtime_manifest

BUCKETS = 32
CORE = ['cohort', 'split', 'query_kind', 'user_id', 'visit_id', 'room_id',
        'prediction_time_ms', 'offset_seconds', 'interval_index', 'label']
BASE_FEATURES = [c for c in MODEL_FEATURES['M_all'] if c not in ROOM + EVENT_FEATURES + CONTENT_META + EMBEDDING_COLUMNS]


def bucket_write(frame, folder, writers, key):
    """Append room-partitioned records; numeric anonymous room IDs are stable."""
    if frame.empty:
        return
    bucket = frame.room_id.astype('int64').to_numpy() % BUCKETS
    for b in np.unique(bucket):
        part = frame.loc[bucket == b].reset_index(drop=True)
        table = pa.Table.from_pandas(part, preserve_index=False)
        name = (key, int(b))
        if name not in writers:
            p = folder / f'{key}_{b:02d}.parquet'; p.parent.mkdir(parents=True, exist_ok=True)
            writers[name] = pq.ParquetWriter(p, table.schema, compression='zstd')
        writers[name].write_table(table)


def close_writers(writers):
    for writer in writers.values():
        writer.close()


def room_inputs(raw, rooms, folder):
    """All released users in target rooms, independent of A/B/C membership."""
    values = pa.array(sorted(rooms), type=pa.string())
    writers = {}; seen = 0
    cols = ['live_id', 'user_id', 'live_play_start_timestamp', 'live_play_end_timestamp']
    try:
        for batch in _csv_batches(raw / 'live_interaction.csv', cols):
            t = pa.Table.from_batches([batch]); mask = pc.is_in(t['live_id'], value_set=values)
            chosen = t.filter(mask)
            if chosen.num_rows:
                frame = chosen.to_pandas()
                frame['raw_row_number'] = pc.filter(pa.array(np.arange(seen, seen + len(t))), mask).to_numpy()
                frame['room_id'] = frame.live_id.astype(str)
                bucket_write(frame, folder, writers, 'audience')
            seen += len(t)
        for kind, filename, roomcol, timecol, milliseconds in [
            ('like', 'live_like.csv', 'live_id', 'like_timestamp', False),
            ('comment', 'live_comment.csv', 'live_stream_id', 'comment_ts', True)]:
            for batch in _csv_batches(raw / filename, [roomcol, 'user_id', timecol]):
                t = pa.Table.from_batches([batch]); t = t.filter(pc.is_in(t[roomcol], value_set=values))
                if not len(t):
                    continue
                frame = t.to_pandas().rename(columns={roomcol: 'room_id'})
                frame['event_ms'] = parse_event_time(frame[timecol], milliseconds=milliseconds)
                frame = frame[frame.event_ms >= 0][['room_id', 'user_id', 'event_ms']]
                bucket_write(frame, folder, writers, kind)
            print(f'raw {kind} extraction complete', flush=True)
        for path in sorted((raw / 'live_emb_128_ts').glob('*.parquet')):
            for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000):
                t = pa.Table.from_batches([batch])
                t = t.filter(pc.is_in(t['live_stream_id'], value_set=values))
                if not len(t):
                    continue
                frame = t.to_pandas().rename(columns={'live_stream_id': 'room_id',
                    'timestamp': 'segment_end_ms', 'feature_128': 'embedding'})
                bucket_write(frame[['room_id', 'segment_id', 'segment_end_ms', 'embedding']],
                             folder, writers, 'content')
            print(f'raw content extraction complete: {path.name}', flush=True)
    finally:
        close_writers(writers)


def base_features(rows, visits):
    result = enrich_points(rows.reset_index(drop=True))
    result = add_visit_history(result, visits)
    for frame in [personal_features(result, visits), local_time_features(result)]:
        result[frame.columns] = frame
    return result


def build_formal_features(cfg=None, *, expected_raw=None, expected_vocabulary=None):
    cfg = cfg or load_config(); root = cfg['data']['run_dir']; raw = cfg['data']['raw_dir']
    repo = Path(__file__).resolve().parents[2]
    if root.exists():
        raise FileExistsError(f'Use a fresh KUAILIVE_RUN_ID; refusing to overwrite {root}')
    expected = expected_raw or json.loads((repo / 'configs/raw_sha256.json').read_text())
    for name, digest in expected['files'].items():
        if not (raw / name).is_file():
            raise FileNotFoundError(f'Missing raw input {name}; run scripts/download_data.py')
        if sha256(raw / name) != digest:
            raise ValueError(f'Raw checksum mismatch: {name}')
    started = time.monotonic(); source = source_manifest(repo)
    root.mkdir(parents=True)
    print("raw checksums verified; selecting cohorts", flush=True)
    selection = select_cohorts(cfg)
    print("cohorts verified; extracting raw visits", flush=True)
    extract_visits(cfg)
    a_train = pd.concat([pd.read_parquet(p) for p in sorted((root / 'visits/A').glob('*.parquet'))], ignore_index=True)
    edges, grid_audit = choose_cutpoints(a_train, cfg)
    # The committed category vocabulary must be independently recoverable from A/train.
    vocab = expected_vocabulary or json.loads((repo / 'configs/entry_category_vocab.json').read_text())
    for field, definition in vocab['fields'].items():
        counts = a_train.loc[a_train.split.eq('train'), field].astype('string').str.strip().value_counts()
        kept = sorted(str(x) for x, n in counts.items() if n >= vocab['minimum_visit_count'])
        if kept != definition['kept_categories']:
            raise ValueError(f'Frozen A/train entry vocabulary does not reproduce: {field}')
    del a_train
    writers = {}; rooms = set(); flow = Counter(); durations = {}
    try:
        for cohort in ['A', 'B', 'C']:
            test_duration = []
            for path in sorted((root / 'visits' / cohort).glob('*.parquet')):
                visits = pd.read_parquet(path)
                rooms.update(visits.room_id.astype(str))
                intervals = make_periods(visits, edges, cfg, only_split=None if cohort == 'A' else 'test')
                intervals['query_kind'] = 'interval'
                h30 = [h30_from_visits(visits, cfg, 'E2-V', only_split='test')]
                if cohort == 'A':
                    h30.append(h30_from_visits(visits, cfg, 'E2-V', only_split='validation'))
                for f in h30:
                    f['query_kind'] = 'H30'
                rows = pd.concat([intervals, *h30], ignore_index=True)
                for (split, kind, status), n in rows.groupby(['split', 'query_kind', 'label_status']).size().items():
                    flow[f'{cohort}|{split}|{kind}|{status}'] += int(n)
                rows = rows[rows.label_status.eq('evaluable') & rows.label.notna()].copy()
                rows['cohort'] = cohort
                result = base_features(rows, visits)
                bucket_write(result[CORE + BASE_FEATURES], root / 'base_features', writers, 'points')
                test = visits[visits.split.eq('test') & visits.quality_ok & visits.end_cause.eq('user')]
                test_duration.extend(((test.session_end_ms - test.session_start_ms) / 1000).tolist())
                print(f'base features complete: {cohort}/{path.name}: {len(result):,} rows', flush=True)
            d = np.array(test_duration)
            durations[cohort] = {'valid_user_exit_visits': len(d),
                **{f'exit_within_{h}s_fraction': float((d <= h).mean()) for h in [5, 30, 35]}}
    finally:
        close_writers(writers)
    room_inputs(raw, rooms, root / 'contexts')
    feature_dir = root / 'model_features'; feature_dir.mkdir()
    shards = []
    for b in range(BUCKETS):
        path = root / 'base_features' / f'points_{b:02d}.parquet'
        if not path.exists():
            continue
        points = pd.read_parquet(path).reset_index(drop=True)
        context = root / 'contexts'
        audience_path = context / f'audience_{b:02d}.parquet'
        raw_room = pd.read_parquet(audience_path) if audience_path.exists() else pd.DataFrame()
        intervals = r2_room_visits(raw_room) if not raw_room.empty else pd.DataFrame()
        by_room = {str(r): g for r, g in intervals.groupby('room_id', sort=False)} if not intervals.empty else {}
        room_out = np.zeros((len(points), len(ROOM)), dtype=np.float32)
        for room, idx in points.groupby('room_id', sort=False).groups.items():
            if str(room) in by_room:
                room_out[idx] = room_features_for_room(points.loc[idx], by_room[str(room)]).to_numpy()
        points[ROOM] = room_out
        del raw_room, intervals, by_room
        for kind in ['like', 'comment']:
            event_path = context / f'{kind}_{b:02d}.parquet'
            events = pd.read_parquet(event_path) if event_path.exists() else pd.DataFrame()
            counts = asof_action_counts(points, events, kind)
            points[counts.columns] = counts
        points = join_content_bucket(points, [context / f'content_{b:02d}.parquet'])
        for c in MODEL_FEATURES['M_all']:
            points[c] = points[c].astype('float32')
        out = feature_dir / f'features_{b:02d}.parquet'
        points[CORE + MODEL_FEATURES['M_all']].to_parquet(out, index=False)
        shards.append({'file': out.name, 'rows': len(points), 'sha256': sha256(out),
                       'row_counts': {f'{a}|{s}|{k}': int(n) for (a,s,k),n in
                                      points.groupby(['cohort','split','query_kind']).size().items()}})
        print(f'formal feature shard complete: {out.name}: {len(points):,}', flush=True)
    manifest = {'run_id': cfg['data']['run_id'], 'data_revision': cfg['data']['revision'],
        'raw_sha256': expected['files'], 'source': source, 'runtime': runtime_manifest(),
        'selection': selection, 'edges_seconds': edges, 'grid_audit': grid_audit,
        'label_flow': dict(flow), 'test_exit_duration': durations,
        'audience_population': 'all released sampled users in target rooms; raw overlapping fragments unioned without gaps',
        'history_availability': 'strictly before t after max(g, delta_minus) seconds',
        'window_convention': '[t-window,t)', 'feature_shards': shards,
        'rows': sum(s['rows'] for s in shards), 'elapsed_seconds': time.monotonic() - started}
    (root / 'feature_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({'status':'completed','rows':manifest['rows'],'edges':edges}, indent=2), flush=True)
    return manifest
