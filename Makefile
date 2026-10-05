.PHONY: test smoke download features formal learning-curve audit report

test:
	python -m pytest -q

smoke:
	PYTHONPATH=src python scripts/run_raw_smoke.py

download:
	PYTHONPATH=src python scripts/download_data.py

features:
	PYTHONPATH=src python scripts/build_features.py

formal:
	PYTHONPATH=src python scripts/run_final_ablation.py --phase train-b
	PYTHONPATH=src python scripts/run_final_ablation.py --phase eval-c

learning-curve:
	PYTHONPATH=src python scripts/run_learning_curve.py

audit:
	PYTHONPATH=src python scripts/run_protocol_audit.py

report:
	PYTHONPATH=src python scripts/write_result_tables.py
