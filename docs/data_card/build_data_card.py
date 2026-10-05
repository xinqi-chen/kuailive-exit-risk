"""Render docs/data_card/data_card.html from stats.json (aggregates only).

Usage: python docs/data_card/build_data_card.py
Field meanings are read from column names or the data unless the KuaiLive-M3
paper (arXiv:2607.24862) documents them.
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "stats.json").read_text())
I, R, L, C, E = S["interaction"], S["room_meta"], S["like"], S["comment"], S["embedding"]
N = I["n_rows"]


def fmt(x, d=0):
    return f"{x:,.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def esc(s):
    return html.escape(str(s))


# ---------------------------------------------------------------- charts
def hbar(rows, label_w=150, width=560, bar_h=16, gap=6, note=None, value_fmt=None):
    """Horizontal single-series bar chart (inline SVG, hover via <title>)."""
    total = sum(n for _, n in rows)
    vmax = max(n for _, n in rows)
    plot_w = width - label_w - 70
    h = len(rows) * (bar_h + gap) + 4
    out = [f'<svg class="chart" viewBox="0 0 {width} {h}" role="img" width="100%">']
    for i, (lab, n) in enumerate(rows):
        y = i * (bar_h + gap) + 2
        w = max(2, plot_w * n / vmax)
        share = n / total
        val = value_fmt(n, share) if value_fmt else pct(share)
        out.append(f'<g><title>{esc(lab)}: {fmt(n)} ({pct(share, 2)})</title>'
                   f'<text x="{label_w - 8}" y="{y + bar_h - 4}" class="lab" text-anchor="end">{esc(lab)}</text>'
                   f'<rect x="{label_w}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="3" class="bar"/>'
                   f'<text x="{label_w + w + 6:.1f}" y="{y + bar_h - 4}" class="val">{val}</text></g>')
    out.append("</svg>")
    if note:
        out.append(f'<p class="note">{note}</p>')
    return "".join(out)


def vbar(rows, width=560, height=170, label_every=1, ylab=""):
    """Vertical single-series bar chart."""
    vmax = max(n for _, n in rows)
    left, bottom, top = 46, 24, 8
    pw, ph = width - left - 6, height - bottom - top
    bw = pw / len(rows)
    out = [f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" width="100%">']
    for frac in (0.5, 1.0):
        y = top + ph * (1 - frac)
        out.append(f'<line x1="{left}" x2="{width - 6}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                   f'<text x="{left - 6}" y="{y + 4:.1f}" class="tick" text-anchor="end">{vmax * frac / 1e6:.1f}M</text>')
    for i, (lab, n) in enumerate(rows):
        x = left + i * bw + 1
        h = max(1, ph * n / vmax)
        out.append(f'<g><title>{esc(lab)}: {fmt(n)}</title>'
                   f'<rect x="{x:.1f}" y="{top + ph - h:.1f}" width="{bw - 2:.1f}" height="{h:.1f}" rx="2" class="bar"/></g>')
        if i % label_every == 0:
            out.append(f'<text x="{x + (bw - 2) / 2:.1f}" y="{height - 8}" class="tick" text-anchor="middle">{esc(lab)}</text>')
    out.append(f'<line x1="{left}" x2="{width - 6}" y1="{top + ph}" y2="{top + ph}" class="axis"/></svg>')
    if ylab:
        out.append(f'<p class="note">{ylab}</p>')
    return "".join(out)


order = ["<1s", "1-3s", "3-5s", "5-10s", "10-30s", "30-60s", "1-5min", "5-30min", ">=30min"]
dur = {r["bin"]: r["n"] for r in I["duration_hist"]}
dur_chart = hbar([(b.replace("-", "–").replace(">=", "≥"), dur[b]) for b in order], label_w=70, width=460, bar_h=18)
day_chart = vbar([(r["p_date"][6:], r["n"]) for r in I["rows_by_p_date"]], label_every=2,
                 ylab="Visit records per partition day (p_date, March 2026, UTC+8). Hover a bar for the exact count.")
hour_chart = vbar([(str(r["h"]), r["n"]) for r in I["local_hour"]], label_every=3,
                  ylab="Visit records by local start hour (UTC+8).")

# ---------------------------------------------------------------- dictionaries
SRC_MEANING = {
    "follow": "Following page (by name). Different from is_follow_enter.",
    "featured_feed": "Featured feed (by name).", "find_feed": "Discover feed (by name).",
    "nearby": "Nearby page (by name).", "other": "Other sources; grouping rule not published.",
    "search": "Search (by name).", "find_tab": "Discover tab (by name).",
    "detail_head": "Unclear; possibly a detail-page header.", "live_more_square": "‘More live’ square (by name).",
    "fanstop_award": "Business code; meaning not documented.", "profile": "Profile page (by name).",
    "live_end": "Stream-ended page (by name).", "live_square": "Live square (by name).",
    "app_out_push": "Push notification related (by name).", "live_jump": "Live-to-live jump (by name).",
    "live_out_push": "Push related (by name).", "home_live_square": "Home live square (by name).",
    "live_push": "Push related (by name).", "featured_live_square": "Featured live square (by name).",
    "kuaishou_coin_task": "Coin-task entry (by name).", "arrow": "Not documented.",
}
ACT_MEANING = {
    "SLIDE_UP_ENTER": "Swiped up into the room.", "CLICK_ENTER": "Tapped into the room.",
    "SLIDE_DOWN_ENTER": "Swiped down into the room.", "OTHER": "Other action; not documented.",
    "UNKNOWN": "Unknown (a string value, not NULL).", "SCHEMA_ENTER": "Probably a deep-link/route jump (inferred).",
    "AUTO_ENTER": "Automatic entry; not identical to is_auto_play.",
    "RETURE_REFER_LIVE": "Probably a return to a referred stream (inferred; source spelling kept).",
}


def cat_table(rows, meaning):
    top = max(r['n'] for r in rows)
    body = "".join(f"<tr><td><code>{esc(r['value'])}</code></td><td class='num'>{fmt(r['n'])}</td>"
                   f"<td class='num'>{pct(r['n'] / N, 2)}</td>"
                   f"<td class='barcell'><span style='width:{max(1, 100 * r['n'] / top):.1f}%'></span></td>"
                   f"<td>{esc(meaning.get(r['value'], ''))}</td></tr>" for r in rows)
    return ("<table><thead><tr><th>Value</th><th class='num'>Records</th><th class='num'>Share</th><th></th>"
            f"<th>Meaning / evidence</th></tr></thead><tbody>{body}</tbody></table>")


def action_row(name, desc):
    a = I["actions"][name]
    return (f"<tr><td><code>{name}</code></td><td>int64</td><td>{desc}</td>"
            f"<td>&gt;0 in {pct(a['share_positive'], 2)} of records; mean {a['mean']:.3f}; max {fmt(a['max_value'])}</td><td>{'Feature (past visits only)' if name in ('like_cnt','comment_cnt','send_gift_cnt') else 'Not used'}</td></tr>")


q = I["play_duration_ms_quantiles"]
d = I["duration_shares"]
pu = I["per_user"]
fl = I["flags"]
rd = R["dur_p25_p50_p75_p90_p99"]

inter_fields = [
    ("user_id", "int64", "Anonymized viewer ID; the released universe is listed in <code>user_id_set.csv</code>.",
     f"{fmt(I['users'])} distinct; 0 NULL", "Grouping; cohort assignment"),
    ("author_id", "int64", "Streamer (creator) ID.", f"{fmt(I['authors'])} distinct", "Per-streamer history"),
    ("live_id", "int64", "One broadcast (room instance). Joins <code>live_room_meta.live_id</code>; a streamer has many.",
     f"{fmt(I['live_ids'])} distinct; {pct(S['coverage']['interaction_room_meta_match'])} matched to room meta (author consistent {pct(S['coverage']['interaction_room_author_match'])})", "Room key"),
    ("live_play_start_timestamp", "timestamp (ms)", "Watch start. Stored without zone; values are UTC (inferred: p_date equals the UTC+8 calendar date).",
     f"{I['start_min']} → {I['end_max']} (UTC) = 4–31 March 2026 local", "Visit reconstruction"),
    ("live_play_end_timestamp", "timestamp (ms)", "Watch end. Records crossing local midnight are cut at <code>23:59:59.999</code> and continue in a new record.",
     "end − start equals play_duration within 5 s for 100% of records", "Labels (future information; never a feature)"),
    ("p_date", "string YYYYMMDD", "Partition date in UTC+8.", f"{I['p_dates']} days, {I['p_date_min']}–{I['p_date_max']}", "—"),
    ("live_source_category", "string", "Page the user came from (see values below).", f"{len(I['live_source_category'])} values; 0 NULL", "Feature (one-hot, entry state)"),
    ("enter_live_action", "string", "How the user entered (see values below).", f"{len(I['enter_live_action'])} values; 0 NULL", "Feature (one-hot, entry state)"),
    ("is_auto_play", "0/1", "Stream started playing automatically.", f"{pct(fl['is_auto_play'])} of records are 1", "Feature"),
    ("is_follow_enter", "0/1", "User already followed the streamer when entering (by name); not the same as coming from the follow page.", f"{pct(fl['is_follow_enter'])} of records are 1", "Feature"),
    ("is_follow_leave", "0/1", "Follow status at leave time.", f"{pct(fl['is_follow_leave'])} of records are 1", "Not used (known only at exit)"),
    ("play_duration", "int64 ms", "Watch duration of the record.",
     f"median {q['p50'] / 1000:.1f} s; P90 {q['p90'] / 1000:.0f} s; P99 {q['p99'] / 60000:.0f} min; max {q['max'] / 1000:,.0f} s (values stop just below 10<sup>7</sup> ms, possibly a cap)", "Labels; history features"),
]
inter_html = "".join(f"<tr><td><code>{n}</code></td><td>{t}</td><td>{m}</td><td>{s}</td><td>{u}</td></tr>" for n, t, m, s, u in inter_fields)
inter_html += action_row("like_cnt", "Likes sent during the record.")
inter_html += action_row("comment_cnt", "Comments posted during the record.")
inter_html += action_row("send_gift_cnt", "Gift-sending actions during the record.")
inter_html += action_row("send_gift_num", "Number of gift items sent.")
inter_html += action_row("follow_author_cnt", "Follow actions on the streamer.")
inter_html += action_row("cancel_follow_author_cnt", "Unfollow actions.")
inter_html += action_row("share_cnt", "Shares of the stream.")
inter_html += action_row("report_live_cnt", "Reports of the stream.")

room_html = "".join(f"<tr><td><code>{n}</code></td><td>{t}</td><td>{m}</td><td>{s}</td><td>{u}</td></tr>" for n, t, m, s, u in [
    ("live_id", "int64", "Broadcast ID (unique key).", f"{fmt(R['live_ids'])} rows = distinct", "Join key"),
    ("author_id", "int64", "Streamer ID.", f"{fmt(R['authors'])} distinct", "Consistency check"),
    ("live_name", "string", "Broadcast title (free text).", f"NULL in {pct(R['name_null'] / R['n_rows'])}", "Not used, not shown"),
    ("start_timestamp", "timestamp (ms, UTC)", "Broadcast start. Starts before the window are kept (earliest " + R['start_min'][:10] + ").", "0 NULL", "Broadcast age feature"),
    ("end_timestamp", "timestamp (ms, UTC)", "Broadcast end <i>T</i><sub>b</sub>. NULL when the stream was still live at data end.",
     f"NULL in {fmt(R['end_null'])} rows ({pct(R['end_null'] / R['n_rows'], 2)}); never before start", "Exit-cause rule (broadcast vs user)"),
    ("live_duration", "double ms", "end − start.", f"median {rd[1] / 3.6e6:.1f} h; P90 {rd[3] / 3.6e6:.1f} h", "—"),
])

like_html = (f"<tr><td><code>user_id</code></td><td>int64</td><td>Liker.</td><td>{fmt(L['users'])} distinct</td><td>Event features</td></tr>"
             f"<tr><td><code>live_id</code></td><td>int64</td><td>Broadcast.</td><td>{fmt(L['live_ids'])} distinct</td><td>Room join</td></tr>"
             f"<tr><td><code>like_timestamp</code></td><td>timestamp (ms, UTC)</td><td>Time of the like event.</td><td>{L['ts_min'][:16]} → {L['ts_max'][:16]}</td><td>Windows [t−w, t)</td></tr>")
comment_html = (f"<tr><td><code>live_stream_id</code></td><td>int64</td><td>Broadcast (same key as <code>live_id</code>).</td><td>{fmt(C['live_ids'])} distinct</td><td>Room join</td></tr>"
                f"<tr><td><code>user_id</code></td><td>int64</td><td>Commenter.</td><td>{fmt(C['users'])} distinct</td><td>Event features</td></tr>"
                f"<tr><td><code>author_id</code></td><td>int64</td><td>Streamer.</td><td>—</td><td>—</td></tr>"
                f"<tr><td><code>content</code></td><td>string</td><td>Comment text (free text, Chinese). {pct(C['emoji_placeholder_share'])} are the placeholder <code>[表情评论]</code> (emoji comment).</td><td>median length {C['content_len_p50']:.0f} chars</td><td>Not used, not shown</td></tr>"
                f"<tr><td><code>comment_ts</code></td><td>int64 epoch ms</td><td>Time of the comment.</td><td>{C['ts_min'][:16]} → {C['ts_max'][:16]} (UTC)</td><td>Windows [t−w, t)</td></tr>")
g = E["gap_s_p10_p50_p90"]
sp = E["segments_per_stream_p50_p90"]
emb_html = (f"<tr><td><code>live_stream_id</code></td><td>string</td><td>Broadcast ID (as text).</td><td>≈{fmt(E['live_streams_approx'])} distinct</td><td>Room join</td></tr>"
            "<tr><td><code>segment_id</code></td><td>string</td><td>Segment identifier.</td><td>—</td><td>Audit</td></tr>"
            "<tr><td><code>feature_128</code></td><td>list&lt;float32&gt;[128]</td><td>Segment embedding from an industrial multimodal LLM over clips, ASR text and metadata, reduced to 128 dims with PCA (paper). Dimensions have no individual meaning.</td><td>always 128 dims</td><td>Content features</td></tr>"
            f"<tr><td><code>timestamp</code></td><td>int64 epoch ms</td><td>Segment time; this project treats it as the moment the segment becomes available.</td><td>{E['ts_min'][:16]} → {E['ts_max'][:16]}; spacing within a stream median {g[1]:.0f} s (P10 {g[0]:.0f} s, P90 {g[2] / 60:.0f} min)*</td><td>Past-only alignment</td></tr>")

head = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KuaiLive-M3 Data Card</title>
<style>
:root{{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e3df;--soft:#f3f2ef;--bar:#2a78d6;--accent:#2a78d6}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#8f8e86;--line:#34332f;--soft:#242422;--bar:#3987e5;--accent:#6aa6ee}}}}
:root[data-theme="dark"]{{--bg:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--muted:#8f8e86;--line:#34332f;--soft:#242422;--bar:#3987e5;--accent:#6aa6ee}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Noto Sans SC",sans-serif}}
main{{max-width:980px;margin:0 auto;padding:32px 16px 64px}}
h1{{font-size:28px;margin:0 0 4px}} h2{{font-size:20px;margin:40px 0 10px;padding-top:12px;border-top:1px solid var(--line)}} h3{{font-size:16px;margin:24px 0 8px}}
.sub{{color:var(--ink2);margin:0 0 20px}} p{{margin:8px 0}} code{{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--soft);padding:1px 4px;border-radius:4px}}
.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:18px 0}}
.tile{{background:var(--soft);border-radius:8px;padding:12px 14px}} .tile b{{display:block;font-size:22px;font-variant-numeric:tabular-nums}} .tile span{{color:var(--ink2);font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:13.5px;margin:8px 0 4px}} th,td{{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--ink2);font-weight:600;font-size:12.5px}} td.barcell{{width:22%;min-width:90px}} td.barcell span{{display:block;height:10px;margin-top:5px;border-radius:0 3px 3px 0;background:var(--bar)}}
td.num,th.num{{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}}
.wrap{{overflow-x:auto}} .note{{color:var(--muted);font-size:12.5px}} .grid2{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}
@media (max-width:760px){{.grid2{{grid-template-columns:1fr}}}}
.chart .bar{{fill:var(--bar)}} .chart .lab,.chart .val,.chart .tick{{fill:var(--ink2);font-size:13px}} .chart .val{{fill:var(--ink)}} .chart .grid{{stroke:var(--line)}} .chart .axis{{stroke:var(--muted)}}
.chart g:hover .bar{{opacity:.75}}
.callout{{border-left:3px solid var(--accent);background:var(--soft);padding:10px 14px;border-radius:0 6px 6px 0;margin:14px 0}}
svg.chart{{max-width:560px}}
@media print{{.grid2{{grid-template-columns:1fr 1fr}} body{{font-size:11.5px}} main{{padding:0}} h2{{break-after:avoid}} table,svg,.tile{{break-inside:avoid}} .wrap{{overflow:visible}}}}
</style></head><body><main>"""

body = f"""
<h1>KuaiLive-M3 live-viewing data card</h1>
<p class="sub">Descriptive statistics and a field dictionary for the raw files this project uses. Source: Hugging Face <code>{S['source']}</code>, revision <code>{S['revision'][:12]}</code>. All numbers are computed over the full released files (not only the project cohorts) by <code>docs/data_card/compute_stats.py</code>. No row-level data are shown.</p>

<div class="callout">The upstream repository has no data card or field-level documentation. The KuaiLive-M3 paper (arXiv:2607.24862) lists the column names and describes user sampling and how the segment embeddings were built. All other meanings below are read from the field names (<i>by name</i>) or derived from the data (<i>inferred</i>), with the supporting statistic next to them.</div>

<div class="tiles">
<div class="tile"><b>{fmt(I['users'])}</b><span>viewers (sampled active users)</span></div>
<div class="tile"><b>{N / 1e6:.1f}M</b><span>visit records</span></div>
<div class="tile"><b>{R['n_rows'] / 1e6:.2f}M</b><span>broadcasts (rooms)</span></div>
<div class="tile"><b>{I['authors'] / 1e6:.2f}M</b><span>streamers</span></div>
<div class="tile"><b>28 days</b><span>4–31 March 2026 (UTC+8)</span></div>
<div class="tile"><b>{pct(d['under_5s'], 0)}</b><span>of records last &lt; 5 s</span></div>
</div>

<h2>1. Files</h2>
<div class="wrap"><table><thead><tr><th>File</th><th class="num">Rows</th><th>One row is</th><th>Used here</th></tr></thead><tbody>
<tr><td><code>live/live_interaction.csv</code></td><td class="num">{fmt(N)}</td><td>One viewing record: a user in one broadcast, with per-record action counts</td><td>Visits, labels, history, entry state</td></tr>
<tr><td><code>live/live_room_meta.parquet</code></td><td class="num">{fmt(R['n_rows'])}</td><td>One broadcast with start/end time</td><td>Broadcast age, broadcast-end rule</td></tr>
<tr><td><code>live/live_like.csv</code></td><td class="num">{fmt(L['n_rows'])}</td><td>One timestamped like event</td><td>Like counts in past windows</td></tr>
<tr><td><code>live/live_comment.csv</code></td><td class="num">{fmt(C['n_rows'])}</td><td>One timestamped comment</td><td>Comment counts in past windows</td></tr>
<tr><td><code>live/live_emb_128_ts/</code> (18 parts)</td><td class="num">{fmt(E['rows'])}</td><td>One content segment of a broadcast with a 128-d embedding</td><td>Latest past content embedding</td></tr>
<tr><td><code>user_id_set.csv</code></td><td class="num">{fmt(S['user_id_set']['rows'])}</td><td>One released user ID</td><td>User universe</td></tr>
</tbody></table></div>
<p class="note">Not used: live impressions, shares, questionnaire, room-level embeddings, author profiles and the short-video domain. Note that event files are not one-to-one with the per-record counters: <code>live_like.csv</code> has {L['n_rows'] / 1e6:.1f}M events while <code>like_cnt</code> sums to {I['actions']['like_cnt']['total'] / 1e6:.1f}M; <code>live_comment.csv</code> has {C['n_rows'] / 1e6:.1f}M rows while <code>comment_cnt</code> sums to {I['actions']['comment_cnt']['total'] / 1e6:.1f}M. Treat them as separate measurements.</p>

<h2>2. Viewing records (<code>live_interaction.csv</code>)</h2>
<h3>Watch duration</h3>
<p>Very short records dominate: median {q['p50'] / 1000:.1f} s, {pct(d['under_5s'])} under 5 s, {pct(d['under_30s'])} under 30 s; only {pct(d['at_least_600s'])} reach 10 min. This is why the project uses a fine time grid in the first seconds.</p>
<div class="grid2"><div>{dur_chart}<p class="note">Share of all {fmt(N)} records in each duration range (unequal widths).</p></div>
<div class="wrap"><table><thead><tr><th>Quantile</th><th class="num">play_duration</th></tr></thead><tbody>
{''.join(f"<tr><td>{k.upper()}</td><td class='num'>{q[k] / 1000:,.1f} s</td></tr>" for k in ['p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'p999', 'max'])}
</tbody></table></div></div>

<h3>Time coverage</h3>
<div class="grid2"><div>{day_chart}</div><div>{hour_chart}</div></div>

<h3>Per viewer</h3>
<p>The median viewer has {pu['visits_p10_p50_p90'][1]:,.0f} records (P10 {pu['visits_p10_p50_p90'][0]:,.0f}, P90 {pu['visits_p10_p50_p90'][2]:,.0f}; max {fmt(pu['visits_max'])}), is active on {pu['active_days_p50']:.0f} of 28 days, watches {pu['authors_p50']:.0f} distinct streamers and {pu['watch_hours_p50_p90'][0]:.0f} h in total (P90 {pu['watch_hours_p50_p90'][1]:.0f} h). Heavy users contribute many records, so record-level statistics are dominated by them.</p>

<h3>Field dictionary</h3>
<div class="wrap"><table><thead><tr><th>Field</th><th>Type</th><th>Meaning</th><th>Statistics</th><th>Use in this project</th></tr></thead><tbody>{inter_html}</tbody></table></div>
<p class="note">No field has NULL values. {fmt(I['exact_duplicate_rows'])} records are exact duplicates of another record. Action counters describe the whole record, so they are used only for <i>completed</i> past visits, never for the current visit.</p>

<h3>Entry values</h3>
<p><b><code>enter_live_action</code></b>: most entries are swipes in the vertical feed or taps.</p>
<div class="wrap">{cat_table(I['enter_live_action'], ACT_MEANING)}</div>
<p><b><code>live_source_category</code></b>: more than half of all records come from the following page.</p>
<div class="wrap">{cat_table(I['live_source_category'], SRC_MEANING)}</div>
<p class="note">The model keeps categories with at least 1,000 visits in cohort A's training period and maps the rest to “other” (vocabulary: <code>configs/entry_category_vocab.json</code>).</p>

<h2>3. Broadcasts (<code>live_room_meta.parquet</code>)</h2>
<div class="wrap"><table><thead><tr><th>Field</th><th>Type</th><th>Meaning</th><th>Statistics</th><th>Use in this project</th></tr></thead><tbody>{room_html}</tbody></table></div>
<p class="note">Every viewing record joins exactly one broadcast with the same streamer. {pct(S['coverage']['visit_ends_after_room_end_5s'], 1)} of records end more than 5 s after their broadcast's recorded end; the sample protocol handles these explicitly.</p>

<h2>4. Like and comment events</h2>
<h3><code>live_like.csv</code></h3>
<div class="wrap"><table><thead><tr><th>Field</th><th>Type</th><th>Meaning</th><th>Statistics</th><th>Use</th></tr></thead><tbody>{like_html}</tbody></table></div>
<p class="note">Median liker has {L['per_user_p50']:,.0f} like events (P90 {L['per_user_p90']:,.0f}).</p>
<h3><code>live_comment.csv</code></h3>
<div class="wrap"><table><thead><tr><th>Field</th><th>Type</th><th>Meaning</th><th>Statistics</th><th>Use</th></tr></thead><tbody>{comment_html}</tbody></table></div>

<h2>5. Content segments (<code>live_emb_128_ts/</code>)</h2>
<div class="wrap"><table><thead><tr><th>Field</th><th>Type</th><th>Meaning</th><th>Statistics</th><th>Use</th></tr></thead><tbody>{emb_html}</tbody></table></div>
<p class="note">* Spacing and segments per stream (median {sp[0]:.0f}, P90 {sp[1]:.0f}) are from a 1% hash sample of streams across all 18 parts; parts are not split by stream. The upstream model and PCA training window are not documented, so the embeddings could contain information from later in the month.</p>

<h2>6. Known caveats</h2>
<ul>
<li><b>Day partitioning.</b> Records crossing local midnight are cut into two records (end <code>23:59:59.999</code>, restart <code>00:00:00.000</code>). Without stitching they create false exits and entries.</li>
<li><b>Overlapping records.</b> One user can have overlapping records in different rooms (e.g. background playback). Which room the user was watching is ambiguous during the overlap.</li>
<li><b>Broadcast end.</b> Records of viewers who stayed to the end close a few seconds before or after the broadcast end, so “record ended” is not always a user decision.</li>
<li><b>Sampled audience.</b> Only ~22k sampled users are released; other viewers in a room are not observed.</li>
<li><b>Duration cap.</b> No record exceeds 10<sup>7</sup> ms (about 2.8 h); longer views may be split or truncated.</li>
</ul>
<p>How the project turns these records into visits and exit labels: <code>docs/sample_protocol.md</code>.</p>

<h2>7. Cohorts used by this project</h2>
<p>All released users are ordered by SHA-256 of their ID. Cohort A (1,000 users) is used for training (first 18 days) and validation (next 5 days). Cohorts B (300) and C (1,000) are disjoint from A and drawn from the 21,748 users with at least one record in the last five days, which is their test period. The project therefore uses 2,300 of the 21,938 users; cohort hashes are recorded in <code>results/formal-v1.1.0/feature_manifest.json</code>.</p>

<p class="note">License: the dataset page lists CC BY-SA 4.0 while the paper states CC BY-NC-SA 4.0; treat as non-commercial research use; cite arXiv:2607.24862. Generated from <code>stats.json</code> (computed in {S['elapsed_seconds']:.0f} s with DuckDB).</p>
</main></body></html>"""

(HERE / "data_card.html").write_text(head + body)
print("wrote", HERE / "data_card.html")
