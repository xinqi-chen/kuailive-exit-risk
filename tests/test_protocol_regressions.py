import numpy as np
import pandas as pd
from kuailive_exit.config import load_config, utc_ms
from kuailive_exit.labels.periods import make_periods, h30_from_visits
from kuailive_exit.features.history import prefix_kaplan_meier, add_visit_history
from kuailive_exit.features.events import asof_action_counts
from kuailive_exit.features.browsing import r2_room_visits, room_features_for_room, attach_visit_actions
from kuailive_exit.features.groups import enrich_points, ENTRY_CATEGORY_FEATURES


def visit():
    t = utc_ms('2026-03-10T00:00:00Z')
    return dict(user_id='1', author_id='2', room_id='3', visit_id='v',
        session_start_ms=t, session_end_ms=t+120000, room_end_ms=t+1000000,
        event_time_ms=t+120000, split='train', quality_ok=True, end_cause='user',
        r1_censor_time_ms=np.nan, overlap_start_times_ms=[], overlap_uncertain_end_ms=-1)


def test_exact_km_counts_fractional_second_events_and_censoring():
    s30, s60 = prefix_kaplan_meier([1500], [True], [0, 1])
    np.testing.assert_array_equal(s30, [1, 0])
    np.testing.assert_array_equal(s60, [1, 0])
    s30, s60 = prefix_kaplan_meier([20123, 30000, 40123], [True, False, True], [3])
    np.testing.assert_allclose(s30, [2/3]); np.testing.assert_allclose(s60, [0])


def test_future_history_does_not_change_exact_km():
    row = visit(); t = row['session_start_ms']
    past = dict(row, session_end_ms=t+1500, event_time_ms=t+1500,
                reported_duration_ms=1500, history_available_ms=t+6500)
    query = pd.DataFrame([dict(row, visit_id='query', prediction_time_ms=t+10000)])
    expected = add_visit_history(query, pd.DataFrame([past]))
    future = dict(past, visit_id='future', session_end_ms=t+15000,
                  history_available_ms=t+20000, reported_duration_ms=15000)
    actual = add_visit_history(query, pd.DataFrame([past, future]))
    np.testing.assert_array_equal(expected.user_survival_30s, actual.user_survival_30s)
    waiting = add_visit_history(query.assign(prediction_time_ms=t+6000), pd.DataFrame([past]))
    assert waiting.user_history_count.iloc[0] == 0


def test_overlap_censors_locally_and_retains_earlier_negatives():
    row = visit(); t = row['session_start_ms']
    row.update(overlap_start_times_ms=[t+90000], overlap_uncertain_end_ms=t+90000)
    result = make_periods(pd.DataFrame([row]), [0, 5, 50, 75, 120], load_config())
    assert result.iloc[0].label_status == 'evaluable'
    assert result.iloc[0].label == 0
    assert result.iloc[2].label_status == 'evaluable'
    assert result.iloc[3].label_status == 'ambiguous_overlap'


def test_room_others_counted_when_query_user_has_no_events_and_rooms_isolated():
    q = pd.DataFrame({'user_id':['1','1'], 'room_id':['r','s'], 'prediction_time_ms':[60000,60000]})
    e = pd.DataFrame({'user_id':['2'], 'room_id':['r'], 'event_ms':[59999]})
    result = asof_action_counts(q, e, 'like')
    assert result.room_other_like_60s.tolist() == [1, 0]
    assert result.own_like_60s.tolist() == [0, 0]


def test_future_reentry_cannot_rewrite_audience_or_exit_counts():
    raw = pd.DataFrame({'raw_row_number':[1,2], 'live_id':['3','3'], 'user_id':['other','other'],
        'live_play_start_timestamp':['1970-01-01T00:00:01Z','1970-01-01T00:00:10.800Z'],
        'live_play_end_timestamp':['1970-01-01T00:00:10Z','1970-01-01T00:00:20Z']})
    q = pd.DataFrame({'user_id':['query'], 'prediction_time_ms':[10500]})
    a = room_features_for_room(q, r2_room_visits(raw.iloc[:1]))
    b = room_features_for_room(q, r2_room_visits(raw))
    np.testing.assert_array_equal(a, b)
    assert b.room_other_exits_60s.iloc[0] == 1


def test_visit_actions_include_midnight_continuation_lineage():
    v = pd.DataFrame({'source_raw_row_numbers':[[10,11]]})
    raw = pd.DataFrame({'raw_row_number':[10,11], 'send_gift_cnt':[0,1],
                        'like_cnt':[0,0], 'comment_cnt':[0,1]})
    out = attach_visit_actions(v, raw)
    assert out.past_visit_has_gift.iloc[0] == 1
    assert out.past_visit_has_comment.iloc[0] == 1


def test_h30_uses_common_identity_and_configured_split():
    row=visit(); row['split']='validation'
    row['session_start_ms']=utc_ms('2026-03-22T00:00:00Z')
    row['session_end_ms']=row['event_time_ms']=row['session_start_ms']+120000
    row['room_end_ms']=row['session_start_ms']+1000000
    out=h30_from_visits(pd.DataFrame([row]),load_config(),'E2-V',only_split='validation')
    assert len(out)==3 and out.interval_index.eq(-1).all()


def test_entry_categories_are_encoded_from_raw_entry_fields():
    q=pd.DataFrame({'offset_seconds':[0], 'prediction_time_ms':[10000], 'room_start_ms':[1000],
        'is_auto_play':['1'], 'is_follow_enter':['0'], 'enter_live_action':['CLICK_ENTER'],
        'live_source_category':['not-in-training']})
    out=enrich_points(q)
    assert out[ENTRY_CATEGORY_FEATURES].to_numpy().sum()==2


def test_empty_history_defaults_do_not_reveal_future_visits():
    row = visit(); t = row['session_start_ms']
    future = dict(row, reported_duration_ms=120000, history_available_ms=t+125000)
    query = pd.DataFrame([dict(row, visit_id='query', prediction_time_ms=t+10000)])
    with_future = add_visit_history(query, pd.DataFrame([future]))
    without_future = add_visit_history(query, pd.DataFrame([future]).iloc[:0])
    pd.testing.assert_frame_equal(with_future, without_future)


def test_multi_midnight_chain_keeps_all_original_action_rows(tmp_path):
    import json
    from kuailive_exit.data.raw import normalize_core
    from kuailive_exit.features.browsing import attach_visit_actions
    cfg = load_config(); cfg['data']['processed_dir'] = tmp_path
    starts = pd.to_datetime(['2026-03-05T15:59:58Z','2026-03-05T16:00:00Z','2026-03-06T16:00:00Z'])
    ends = pd.to_datetime(['2026-03-05T15:59:59.999Z','2026-03-06T15:59:59.999Z','2026-03-06T16:00:01Z'], format='mixed')
    raw = pd.DataFrame({'user_id':['1']*3,'author_id':['2']*3,'live_id':['3']*3,
        'live_play_start_timestamp':starts.astype(str),'live_play_end_timestamp':ends.astype(str),
        'raw_row_number':[10,11,12],'p_date':['20260305','20260306','20260307'],
        'play_duration':((ends-starts).total_seconds()*1000).astype(str),
        'is_follow_leave':['0']*3,'send_gift_cnt':[0,0,1],'like_cnt':[0]*3,'comment_cnt':[0,1,0]})
    raw.to_parquet(tmp_path/'selected_interactions.parquet',index=False)
    pd.DataFrame({'live_id':[3],'author_id':[2], 'start_timestamp':[int(starts[0].timestamp()*1000)-10000],
        'end_timestamp':[int(ends[-1].timestamp()*1000)+10000]}).to_parquet(tmp_path/'selected_rooms.parquet',index=False)
    (tmp_path/'extraction_audit.json').write_text(json.dumps({'data_revision':cfg['data']['revision']}))
    sessions,_ = normalize_core(cfg)
    assert len(sessions)==1
    assert list(sessions.source_raw_row_numbers.iloc[0])==[10,11,12]
    assert sessions.session_end_ms.iloc[0]==int(ends[-1].timestamp()*1000)
    actions = attach_visit_actions(sessions, raw)
    assert actions.past_visit_has_gift.iloc[0]==actions.past_visit_has_comment.iloc[0]==1
