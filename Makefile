# Usage: make data | test | run EXP=<name> | eval EXP=<name> | tables | smoke
PY ?= python
EXP ?= b0_prior
CFG = configs/exp/$(EXP).yaml
RUNS ?= runs

.PHONY: data test run train predict eval reeval paired label-oracle verify tables failures smoke synthetic clean-smoke

data:                      ## raw -> data/processed + label_check.json (fails on bad labels)
	$(PY) scripts/build_perception.py --config configs/exp/b0_prior.yaml --report_templates

test:
	$(PY) -m pytest -q

train:                     ## LoRA fine-tuning (only for configs with train.enabled: true)
	$(PY) scripts/train.py --config $(CFG)

predict:
	$(PY) scripts/predict.py --config $(CFG)

eval:                      ## evaluate the latest run of EXP
	$(PY) scripts/eval_perception.py --run $$($(PY) scripts/_common.py latest $(RUNS) $(EXP))

run:                       ## train (if enabled) + predict + eval
	@if grep -q "enabled: true" $(CFG); then $(MAKE) train EXP=$(EXP); fi
	$(MAKE) predict EXP=$(EXP)
	$(MAKE) eval EXP=$(EXP)

failures:
	$(PY) scripts/viz_failures.py --run $$($(PY) scripts/_common.py latest $(RUNS) $(EXP)) --n 10

verify:                    ## re-check EVERYTHING once: tests, run integrity, re-score, paired CIs, label oracle, tables, report numbers
	$(PY) -m pytest -q
	$(PY) scripts/check_runs.py
	$(MAKE) reeval
	$(MAKE) paired
	$(MAKE) label-oracle
	$(MAKE) tables
	$(PY) scripts/verify_report.py

label-oracle:              ## do turn labels follow past or future heading change? (CPU) -> results/label_oracle.md
	$(PY) scripts/label_oracle.py

paired:                    ## paired comparisons with scene-bootstrap CIs (CPU) -> results/paired.md
	$(PY) scripts/compare_runs.py --runs_dir $(RUNS)

reeval:                    ## re-score every run that was already scored (CPU only; adds 95% CIs)
	for d in $(RUNS)/*/; do if [ -f $$d/_preds.jsonl ] && [ -f $$d/_report.json ]; then $(PY) scripts/eval_perception.py --run $$d || exit 1; fi; done

tables:
	$(PY) scripts/make_tables.py --runs_dir $(RUNS) --out results/tables.md

# ---- smoke test on synthetic data (no GPU, no downloads). Numbers are NOT results.
synthetic:
	$(PY) scripts/make_synthetic.py --out data/synthetic --scenes 24 --samples 6

smoke: synthetic
	$(PY) scripts/build_perception.py --config configs/exp/smoke_mock.yaml
	for e in smoke_prior smoke_mock; do \
	  R=$$($(PY) scripts/predict.py --config configs/exp/$$e.yaml | tail -1); \
	  $(PY) scripts/eval_perception.py --run $$R; done
	$(PY) scripts/make_tables.py --runs_dir runs_smoke --out results/smoke/tables.md --title "SMOKE TEST on synthetic data (not results)"
	$(PY) scripts/viz_failures.py --run $$($(PY) scripts/_common.py latest runs_smoke smoke_mock) --n 6 --out results/smoke/failures

clean-smoke:
	rm -rf data/synthetic runs_smoke results/smoke
