import numpy as np
import pandas as pd

from kuailive_exit.features.content import join_content_bucket
from kuailive_exit.features.events import asof_action_counts
from kuailive_exit.features.browsing import add_room_audience, personal_features


def test_event_windows_are_left_closed_and_right_open():
    points = pd.DataFrame({"user_id": ["u1"], "prediction_time_ms": [60_000]})
    events = pd.DataFrame({"user_id": ["u1", "u2", "u2"],
                           "event_ms": [0, 59_999, 60_000]})

    result = asof_action_counts(points, events, "like").iloc[0]

    assert result["own_like_60s"] == 1
    assert result["room_other_like_60s"] == 1


def test_future_events_do_not_change_features():
    points = pd.DataFrame({"user_id": ["u1"], "prediction_time_ms": [60_000]})
    before = pd.DataFrame({"user_id": ["u1"], "event_ms": [59_000]})
    after = pd.concat([before, pd.DataFrame({"user_id": ["u1"], "event_ms": [60_001]})],
                      ignore_index=True)

    expected = asof_action_counts(points, before, "comment")
    actual = asof_action_counts(points, after, "comment")

    np.testing.assert_array_equal(expected.to_numpy(), actual.to_numpy())


def test_content_asof_excludes_current_and_future_segments(tmp_path):
    points = pd.DataFrame({"room_id": ["r1"], "prediction_time_ms": [20_000]})
    segments = pd.DataFrame({
        "room_id": ["r1", "r1"], "segment_id": ["past", "at_t"],
        "segment_end_ms": [19_999, 20_000],
        "embedding": [np.ones(128, dtype=np.float32), np.full(128, 2, dtype=np.float32)],
    })
    path = tmp_path / "segments.parquet"
    segments.to_parquet(path, index=False)

    result = join_content_bucket(points, [path])

    assert result.loc[0, "has_past_embedding"] == 1
    assert result.loc[0, "embedding_age_seconds"] == 0.001
    assert result.loc[0, "emb_000"] == 1.0


def test_browsing_windows_include_the_exact_left_boundary():
    points = pd.DataFrame({"user_id": ["u1"], "prediction_time_ms": [120_000],
                           "session_start_ms": [119_000]})
    visits = pd.DataFrame({
        "user_id": ["u1", "u1", "u1"], "author_id": ["a", "a", "a"],
        "visit_id": ["v1", "v2", "v3"], "session_start_ms": [0, 60_000, 120_000],
        "session_end_ms": [60_000, 60_001, 120_000], "reported_duration_ms": [60_000, 1, 0],
        "quality_ok": [True, True, True], "end_cause": ["user", "user", "user"],
        "past_visit_has_gift": [0, 0, 0], "past_visit_has_like": [0, 0, 0],
        "past_visit_has_comment": [0, 0, 0],
    })

    result = personal_features(points, visits)

    assert result.loc[0, "browse_starts_60s"] == 1
    assert result.loc[0, "browse_ends_60s"] == 2


def test_room_entry_at_exact_left_boundary_is_counted():
    points = pd.DataFrame({"room_id": ["r1"], "user_id": ["u_query"],
                           "prediction_time_ms": [60_001]})
    raw = pd.DataFrame({"raw_row_number": [1, 2], "live_id": ["r1", "r1"],
                        "user_id": ["u_other", "u_later"],
                        "live_play_start_timestamp": ["1970-01-01T00:00:00.001Z",
                                                      "1970-01-01T00:01:00.001Z"],
                        "live_play_end_timestamp": ["1970-01-01T00:00:01.000Z",
                                                    "1970-01-01T00:01:00.500Z"]})

    result = add_room_audience(points, raw)

    assert result.loc[0, "room_other_entries_60s"] == 1
