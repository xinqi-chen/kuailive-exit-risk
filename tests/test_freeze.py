import importlib.util
import json
from pathlib import Path
import pytest
from kuailive_exit.provenance import sha256


def evaluator(tmp_path, monkeypatch):
    path=Path(__file__).resolve().parents[1]/'scripts/run_final_ablation.py'
    spec=importlib.util.spec_from_file_location('freeze_check',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    exp=tmp_path/'run';result=tmp_path/'results'
    (exp/'models').mkdir(parents=True);result.mkdir()
    for p in [exp/'feature_manifest.json',exp/'training_manifest.json',result/'metrics_long.csv']:
        p.write_text('{}')
    model=exp/'models/M0_fixed.joblib';model.write_bytes(b'frozen-model')
    freeze={'source':{'sha256':'same'},'feature_manifest_sha256':sha256(exp/'feature_manifest.json'),
        'training_manifest_sha256':sha256(exp/'training_manifest.json'),
        'models':{'M0_fixed':{'sha256':sha256(model)}},'input_feature_shards_sha256':{},
        'B_metrics_sha256':sha256(result/'metrics_long.csv')}
    (result/'pre_c_freeze_manifest.json').write_text(json.dumps(freeze))
    monkeypatch.setattr(module,'EXP',exp);monkeypatch.setattr(module,'RESULT',result)
    monkeypatch.setattr(module,'source_manifest',lambda root:{'sha256':'same'})
    return module,exp,result


def test_changed_model_is_rejected_before_any_C_evaluation(tmp_path,monkeypatch):
    module,exp,result=evaluator(tmp_path,monkeypatch)
    (exp/'models/M0_fixed.joblib').write_bytes(b'changed-model')
    with pytest.raises(RuntimeError,match='Model changed'):
        module.eval_c()
    assert not (result/'C_evaluation_started.json').exists()


def test_uncommitted_source_change_is_rejected_before_C(tmp_path,monkeypatch):
    module,exp,result=evaluator(tmp_path,monkeypatch)
    monkeypatch.setattr(module,'source_manifest',lambda root:{'sha256':'changed'})
    with pytest.raises(RuntimeError,match='source/config changed'):
        module.eval_c()
    assert not (result/'C_evaluation_started.json').exists()
