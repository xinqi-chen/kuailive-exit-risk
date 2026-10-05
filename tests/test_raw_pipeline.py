"""A synthetic raw dataset proves that integration needs no private artifacts."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

from kuailive_exit.config import load_config
from kuailive_exit.data.raw import INTERACTION_COLUMNS
from kuailive_exit.pipeline import build_formal_features
from kuailive_exit.provenance import sha256
from kuailive_exit.evaluation.uncertainty import paired_user_bootstrap


def synthetic_raw(tmp_path, repeats=100):
    raw=tmp_path/'raw';raw.mkdir()
    users=[str(i) for i in range(1,25)]
    pd.DataFrame({'user_id':users}).to_csv(raw/'user_id_set.csv',index=False)
    interactions=[];rooms=[];room=100
    for u in users:
        for date,n in [('2026-03-05T00:00:00Z',repeats),('2026-03-23T00:00:00Z',4),('2026-03-28T00:00:00Z',4)]:
            for j in range(n):
                start=pd.Timestamp(date)+pd.Timedelta(seconds=j*1000)
                duration=[1500,5123,20123,90123,150123][j%5]
                end=start+pd.Timedelta(milliseconds=duration);room+=1
                interactions.append(dict(user_id=u,author_id='9',live_id=str(room),
                    live_play_start_timestamp=start.isoformat(),live_play_end_timestamp=end.isoformat(),
                    p_date=start.strftime('%Y%m%d'),live_source_category='follow',
                    enter_live_action='CLICK_ENTER',is_auto_play='0',is_follow_enter='1',
                    is_follow_leave='1',play_duration=str(duration),send_gift_cnt='0',like_cnt='0',comment_cnt='0'))
                rooms.append(dict(live_id=room,author_id=9,start_timestamp=int(start.timestamp()*1000)-1000,
                    end_timestamp=int(start.timestamp()*1000)+10000000,live_duration=10000001))
    pd.DataFrame(interactions)[INTERACTION_COLUMNS].to_csv(raw/'live_interaction.csv',index=False)
    pd.DataFrame(rooms).to_parquet(raw/'live_room_meta.parquet',index=False)
    pd.DataFrame({'user_id':['24'],'live_id':[str(room)],'like_timestamp':['2026-03-28T00:00:00Z']}).to_csv(raw/'live_like.csv',index=False)
    pd.DataFrame({'user_id':['24'],'live_stream_id':[str(room)],'comment_ts':[int(pd.Timestamp('2026-03-28T00:00:00Z').timestamp()*1000)]}).to_csv(raw/'live_comment.csv',index=False)
    (raw/'live_emb_128_ts').mkdir()
    pd.DataFrame({'live_stream_id':[str(room)],'segment_id':['s'],
        'timestamp':[int(pd.Timestamp('2026-03-28T00:00:00Z').timestamp()*1000)-100],
        'feature_128':[np.ones(128,dtype=np.float32)]}).to_parquet(raw/'live_emb_128_ts/part.parquet',index=False)
    return raw,users


def test_smoke_finishes_without_formal_feature_directory(tmp_path):
    raw,_=synthetic_raw(tmp_path,repeats=4)
    repo=Path(__file__).resolve().parents[1]
    env=dict(os.environ,KUAILIVE_DATA_DIR=str(raw),KUAILIVE_PROCESSED_DIR=str(tmp_path/'processed'),
             PYTHONPATH=str(repo/'src'),MPLCONFIGDIR=str(tmp_path/'mpl'))
    output=tmp_path/'smoke'
    result=subprocess.run([sys.executable,str(repo/'scripts/run_raw_smoke.py'),'--output',str(output)],
                          env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    manifest=json.loads((output/'smoke_manifest.json').read_text())
    assert manifest['status']=='completed'
    assert manifest['feature_consistency']['status']=='not_requested'
    predictions=pd.read_parquet(output/'smoke_predictions.parquet')
    assert predictions.prediction.between(0,1).all()


def test_formal_builder_from_raw_including_events_and_content(tmp_path,monkeypatch):
    raw,users=synthetic_raw(tmp_path)
    cfg=copy.deepcopy(load_config())
    root=tmp_path/'formal'
    cfg['data'].update(raw_dir=raw,processed_dir=tmp_path,run_dir=root,run_id='fixture')
    order=sorted(users,key=lambda u:(hashlib.sha256(u.encode()).digest(),u))
    b=order[4:8];a=[u for u in order if u not in b][:16];c=[u for u in order if u not in set(a+b)]
    cfg['cohorts'].update(train_users=4,A_users=16,B_users=4,C_users=4,
        expected_selected_user_ids_sha256={k:hashlib.sha256('\n'.join(v).encode()).hexdigest() for k,v in [('A',a),('B',b),('C',c)]})
    fixture_vocab={'minimum_visit_count':1,'fields':{
        'enter_live_action':{'kept_categories':['CLICK_ENTER']},
        'live_source_category':{'kept_categories':['follow']}}}
    paths=[*raw.glob('*.csv'),*raw.glob('*.parquet'),*raw.glob('live_emb_128_ts/*.parquet')]
    expected={'files':{str(p.relative_to(raw)):sha256(p) for p in paths}}
    manifest=build_formal_features(cfg,expected_raw=expected,expected_vocabulary=fixture_vocab)
    assert manifest['rows']>0
    assert manifest['selection']['cohorts']['C']['users']==4
    tables=[pd.read_parquet(p) for p in (root/'model_features').glob('*.parquet')]
    f=pd.concat(tables,ignore_index=True)
    assert set(f.cohort)=={'A','B','C'}
    assert f.user_survival_30s.min()<1
    assert f.has_past_embedding.max()==1
    assert f[f.query_kind.eq('H30')].interval_index.eq(-1).all()
    assert (f[f.query_kind.eq('H30') & f.cohort.eq('A')].split=='test').any()
    # Exercise public evaluation selection and paired prediction persistence too.
    import importlib.util
    from kuailive_exit.features.groups import MODEL_FEATURES
    script = Path(__file__).resolve().parents[1] / 'scripts/run_final_ablation.py'
    spec = importlib.util.spec_from_file_location('raw_evaluator', script)
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    monkeypatch.setattr(evaluator, 'EXP', root)
    monkeypatch.setattr(evaluator, 'EDGES', manifest['edges_seconds'])
    class ConstantModel:
        def predict_proba(self, x):
            return np.tile([.9, .1], (len(x), 1))
    records = evaluator.metric_records('fixture', ConstantModel(), MODEL_FEATURES['M1'], 'B', 'test')
    predictions = pd.read_parquet(root / 'predictions/B_fixture.parquet')
    assert len(predictions) == records[0]['n']
    assert predictions[['user_id','visit_id','prediction_time_ms','offset_seconds']].duplicated().sum() == 0
    assert predictions.probability.between(0,1).all()


def test_paired_bootstrap_identical_predictions_has_zero_interval():
    out=paired_user_bootstrap(['a','a','b'],[0,1,0],[.1,.8,.2],[.1,.8,.2],resamples=200)
    assert out['log_loss']['difference']==0
    assert out['log_loss']['ci95_lower']==out['log_loss']['ci95_upper']==0
