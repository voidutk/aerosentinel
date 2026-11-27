---
name: run-experiment
description: Run a named AeroSentinel experiment end to end (train over seeds, evaluate, write results JSON, refresh tables). Use when asked to run, rerun or compare experiments.
disable-model-invocation: true
---
Run experiment: $ARGUMENTS

Experiment names follow `<dataset>_<model>_<setting>` (SPEC §4), and the results file is `reports/results/<name>.json`.

1. Confirm `configs/experiment/$ARGUMENTS.yaml` exists and its name matches the naming scheme. If not, list available experiments and stop.
2. Print the resolved config (`uv run python -m aerosentinel.train experiment=$ARGUMENTS --cfg job`) and check:
   - inputs are W + X_s (+ residuals) only
   - the split is by unit
   - the RUL cap matches SPEC §1.3
   - on N-CMAPSS, test stride equals train stride and windows stay within one (unit, cycle)
   - any tuning or calibration step consumes out-of-fold predictions
3. Train over seeds 0, 1, 2: `uv run python -m aerosentinel.train experiment=$ARGUMENTS seed=<s>`. On a CPU-only machine with an N-CMAPSS config at `downsample_factor: 1` (full resolution), stop and suggest `scripts/kaggle_train.sh $ARGUMENTS` instead.
4. Evaluate: `uv run python -m aerosentinel.evaluate experiment=$ARGUMENTS` → `reports/results/$ARGUMENTS.json` with mean ± std (N-CMAPSS: per-cycle primary, per-window secondary; C-MAPSS: capped primary, uncapped secondary).
5. Refresh tables: `uv run python scripts/make_tables.py`.
6. Report:
   - the metrics table
   - the diff vs the previous JSON for this experiment if one exists (git diff)
   - anything suspicious, e.g. capped C-MAPSS FD001 RMSE < 11, coverage far from the 90% target, or many cycles without a valid window
