import numpy as np
import pandas as pd
from kuailive_exit.models.survival import forecast_h30


def test_shuffled_clock_stays_shuffled_during_h30_conversion():
    class Recorder:
        def __init__(self): self.seen = []
        def predict_proba(self, x):
            self.seen.append(x.copy())
            return np.tile([.9, .1], (len(x), 1))
    points = pd.DataFrame({'offset_seconds':[30], 'prediction_time_ms':[0],
        'room_elapsed_seconds':[40], 'browse_active_segment_seconds':[8],
        'browse_last_start_age_seconds':[9], 'browse_local_hour':[17],
        'browse_local_weekday':[6]})
    columns = ['elapsed_watch_seconds','browse_active_segment_seconds','browse_local_hour']
    model = Recorder()
    forecast_h30(model, points, [0, 35, 60, 600], columns, 'placebo',
                 frozen_columns=['browse_local_hour'])
    assert all(x[0,2] == 17 for x in model.seen)
    assert [x[0,0] for x in model.seen] == [0, 35]
