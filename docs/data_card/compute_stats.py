"""Compute descriptive statistics of the raw KuaiLive-M3 files used by this project.

Usage: python docs/data_card/compute_stats.py  (reads $KUAILIVE_DATA_DIR or data/raw; needs duckdb)
Writes docs/data_card/stats.json. Aggregates only; no row-level values are written.
"""
import json, os, time
from pathlib import Path
import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("KUAILIVE_DATA_DIR", ROOT / "data/raw"))
OUT = Path(__file__).with_name("stats.json")
con = duckdb.connect()
con.execute("SET memory_limit='1GB'; SET threads=2; SET preserve_insertion_order=false")
con.execute(f"SET temp_directory='{Path.home() / 'duckdb_tmp'}'")
t0 = time.time()
S = {"source": "imgkkk2004/KuaiLive-M3", "revision": "417923d442a793b3e92cc7e079dd4b73fe706c30"}
def q(sql): return con.execute(sql).fetchall()
def q1(sql): return con.execute(sql).fetchone()
def rec(sql):
    cur = con.execute(sql); cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]

I = f"read_csv('{RAW/'live_interaction.csv'}', header=true, types={{'live_play_start_timestamp':'TIMESTAMP','live_play_end_timestamp':'TIMESTAMP','p_date':'VARCHAR','live_source_category':'VARCHAR','enter_live_action':'VARCHAR'}})"
CACHE = Path.home() / "inter_cache.parquet"
if not CACHE.exists():
    con.execute(f"COPY (SELECT * FROM {I}) TO '{CACHE}' (FORMAT parquet, ROW_GROUP_SIZE 500000)")
con.execute(f"CREATE TEMP VIEW inter AS SELECT * FROM read_parquet('{CACHE}')")
cols = [r[0] for r in q("DESCRIBE inter")]
types = {r[0]: r[1] for r in q("DESCRIBE inter")}
S["interaction"] = {"columns": cols, "types": types}
it = S["interaction"]
it.update(rec("""SELECT count(*) AS n_rows, count(DISTINCT user_id) users, count(DISTINCT author_id) authors,
 count(DISTINCT live_id) live_ids, min(live_play_start_timestamp)::VARCHAR start_min, max(live_play_end_timestamp)::VARCHAR end_max,
 min(p_date) p_date_min, max(p_date) p_date_max, count(DISTINCT p_date) p_dates FROM inter""")[0])
it["nulls"] = {c: q1(f'SELECT count(*) - count("{c}") FROM inter')[0] for c in cols}
it["play_duration_ms_quantiles"] = dict(zip(["p01","p05","p10","p25","p50","p75","p90","p95","p99","p999","max"],
  [*q1("SELECT quantile_cont(play_duration,[0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.999]) FROM inter")[0], q1("SELECT max(play_duration) FROM inter")[0]]))
it["duration_shares"] = rec("""SELECT avg((play_duration<=0)::INT) nonpositive, avg((play_duration<5000)::INT) under_5s, avg((play_duration<30000)::INT) under_30s,
 avg((play_duration>=60000)::INT) at_least_60s, avg((play_duration>=600000)::INT) at_least_600s, avg((play_duration>=3600000)::INT) at_least_1h,
 avg((abs(epoch_ms(live_play_end_timestamp)-epoch_ms(live_play_start_timestamp)-play_duration)>5000)::INT) endpoint_mismatch_over_5s FROM inter""")[0]
it["duration_hist"] = rec("""SELECT CASE WHEN play_duration<1000 THEN '<1s' WHEN play_duration<3000 THEN '1-3s' WHEN play_duration<5000 THEN '3-5s'
 WHEN play_duration<10000 THEN '5-10s' WHEN play_duration<30000 THEN '10-30s' WHEN play_duration<60000 THEN '30-60s'
 WHEN play_duration<300000 THEN '1-5min' WHEN play_duration<1800000 THEN '5-30min' ELSE '>=30min' END AS bin, count(*) n FROM inter GROUP BY 1""")
for c in ["live_source_category", "enter_live_action"]:
    it[c] = rec(f"SELECT coalesce({c},'<NULL>') AS value, count(*) n FROM inter GROUP BY 1 ORDER BY n DESC")
it["flags"] = rec("SELECT avg(is_auto_play) is_auto_play, avg(is_follow_enter) is_follow_enter, avg(is_follow_leave) is_follow_leave FROM inter")[0]
acts = ["like_cnt","comment_cnt","send_gift_cnt","send_gift_num","follow_author_cnt","cancel_follow_author_cnt","share_cnt","report_live_cnt"]
it["actions"] = {a: rec(f"SELECT avg(({a}>0)::INT) share_positive, avg({a}) mean, quantile_cont({a},0.99) p99, max({a}) AS max_value, sum({a}) total FROM inter")[0] for a in acts}
it["rows_by_p_date"] = rec("SELECT p_date, count(*) n FROM inter GROUP BY 1 ORDER BY 1")
it["local_hour"] = rec("SELECT hour(live_play_start_timestamp + INTERVAL 8 HOUR) h, count(*) n FROM inter GROUP BY 1 ORDER BY 1")
it["per_user"] = rec("""WITH u AS (SELECT user_id, count(*) n, count(DISTINCT p_date) d, count(DISTINCT author_id) a, sum(play_duration)/3.6e6 AS watch_h FROM inter GROUP BY 1)
 SELECT quantile_cont(n,[0.1,0.5,0.9]) visits_p10_p50_p90, max(n) AS visits_max,
 quantile_cont(d,0.5) active_days_p50, quantile_cont(a,0.5) authors_p50, quantile_cont(watch_h,[0.5,0.9]) watch_hours_p50_p90 FROM u""")[0]
it["exact_duplicate_rows"] = q1("SELECT count(*) - count(DISTINCT hash(" + ",".join(f'"{c}"' for c in cols) + ")) FROM inter")[0]  # hash-based; collisions negligible
print("interaction done", time.time()-t0, flush=True)
R = f"read_parquet('{RAW/'live_room_meta.parquet'}')"
con.execute(f"CREATE TEMP VIEW room AS SELECT * FROM {R}")
S["room_meta"] = {"types": {r[0]: r[1] for r in q("DESCRIBE room")}}
S["room_meta"].update(rec("""SELECT count(*) AS n_rows, count(DISTINCT live_id) live_ids, count(DISTINCT author_id) authors,
 sum((end_timestamp IS NULL)::INT) end_null, sum((start_timestamp IS NULL)::INT) start_null, sum((live_name IS NULL)::INT) name_null,
 sum((end_timestamp < start_timestamp)::INT) end_before_start, min(start_timestamp)::VARCHAR start_min, max(start_timestamp)::VARCHAR start_max,
 max(end_timestamp)::VARCHAR end_max,
 quantile_cont(live_duration,[0.25,0.5,0.75,0.9,0.99]) dur_p25_p50_p75_p90_p99, max(live_duration) AS dur_max,
 quantile_cont(epoch_ms(end_timestamp)-epoch_ms(start_timestamp),0.5)/1000 end_minus_start_p50_s FROM room""")[0])
S["coverage"] = rec("""SELECT avg((r.live_id IS NOT NULL)::INT) interaction_room_meta_match,
 avg((r.live_id IS NOT NULL AND r.author_id=i.author_id)::INT) interaction_room_author_match,
 avg((r.end_timestamp IS NOT NULL AND i.live_play_end_timestamp > r.end_timestamp + INTERVAL 5 SECOND)::INT) visit_ends_after_room_end_5s
 FROM inter i LEFT JOIN (SELECT DISTINCT ON (live_id) * FROM room) r USING (live_id)""")[0]
print("room done", time.time()-t0, flush=True)
L = f"read_csv('{RAW/'live_like.csv'}', header=true, types={{'like_timestamp':'TIMESTAMP'}})"
S["like"] = rec(f"""SELECT count(*) AS n_rows, count(DISTINCT user_id) users, count(DISTINCT live_id) live_ids,
 min(like_timestamp)::VARCHAR ts_min, max(like_timestamp)::VARCHAR ts_max, sum((like_timestamp IS NULL)::INT) ts_null FROM {L}""")[0]
S["like"]["per_user_p50"], S["like"]["per_user_p90"] = q1(f"WITH u AS (SELECT user_id, count(*) n FROM {L} GROUP BY 1) SELECT quantile_cont(n,0.5), quantile_cont(n,0.9) FROM u")
print("like done", time.time()-t0, flush=True)
C = f"read_csv('{RAW/'live_comment.csv'}', header=true, quote='\"', escape='\"', strict_mode=false, ignore_errors=true)"
S["comment"] = rec(f"""SELECT count(*) AS n_rows, count(DISTINCT user_id) users, count(DISTINCT live_stream_id) live_ids,
 make_timestamp(min(comment_ts)*1000)::VARCHAR ts_min, make_timestamp(max(comment_ts)*1000)::VARCHAR ts_max,
 avg((content='[表情评论]')::INT) emoji_placeholder_share, quantile_cont(length(content),0.5) content_len_p50,
 sum((content IS NULL)::INT) content_null FROM {C}""")[0]
S["comment"]["lines_in_file"] = sum(1 for _ in open(RAW/'live_comment.csv', 'rb')) - 1
print("comment done", time.time()-t0, flush=True)
parts = sorted((RAW/'live_emb_128_ts').glob('*.parquet'))
S["embedding"] = {"parts": len(parts), "rows": sum(pq.ParquetFile(p).metadata.num_rows for p in parts),
  "types": {f.name: str(f.type) for f in pq.ParquetFile(parts[0]).schema_arrow}}
E = "read_parquet([" + ",".join(f"'{p}'" for p in parts) + "])"
S["embedding"].update(rec(f"""SELECT approx_count_distinct(live_stream_id) live_streams_approx,
 make_timestamp(min(timestamp)*1000)::VARCHAR ts_min, make_timestamp(max(timestamp)*1000)::VARCHAR ts_max FROM {E}""")[0])
# 1% stream sample (hash of live_stream_id) across all 18 parts: segments are not partitioned by stream.
S["embedding"]["sample"] = "1% of live_stream_id by hash, all parts"
S["embedding"].update(rec(f"""WITH x AS (SELECT live_stream_id, timestamp, len(feature_128) d FROM {E} WHERE hash(live_stream_id) % 100 = 0),
 s AS (SELECT *, timestamp - lag(timestamp) OVER (PARTITION BY live_stream_id ORDER BY timestamp) gap FROM x),
 c AS (SELECT live_stream_id, count(*) n FROM x GROUP BY 1)
 SELECT min(d) dim_min, max(d) dim_max, quantile_cont(gap/1000,[0.1,0.5,0.9]) gap_s_p10_p50_p90,
   avg((gap=0)::INT) duplicate_timestamp_share, (SELECT quantile_cont(n,[0.5,0.9]) FROM c) segments_per_stream_p50_p90,
   (SELECT max(n) FROM c) segments_per_stream_max FROM s""")[0])
print("emb done", time.time()-t0, flush=True)
S["user_id_set"] = {"rows": q1(f"SELECT count(*) FROM read_csv('{RAW/'user_id_set.csv'}')")[0]}
S["elapsed_seconds"] = time.time()-t0
OUT.write_text(json.dumps(S, indent=1, default=str, ensure_ascii=False))
print("wrote", OUT, flush=True)
