# Build Phases — copy-paste prompts for Claude Code

Each phase ends **shippable**: tests green, README updated, pushed, tagged. Even if you stop after Phase 2, the repo is a finished, presentable project — not a half-built one.

How to run each phase:
1. `/clear` (fresh context per phase).
2. Press `Shift+Tab` until plan mode is on (or start with `claude --permission-mode plan`). Paste the phase prompt. Read the plan; edit it with `Ctrl+G` if needed; approve.
3. Let it implement. When it says done, ask for the evidence (test output, metrics JSON).
4. Run the review prompt at the bottom of the phase (uses the `ml-reviewer` subagent).
5. Commit via PR, tag the release, tick the phase below.

Status: update the checkbox when a phase ships.

- [ ] Phase 0 — Scaffold (½ day) → no tag
- [ ] Phase 1 — C-MAPSS baselines (2–3 days) → `v0.1`
- [ ] Phase 2 — N-CMAPSS pipeline + EDA (2–3 days) → `v0.2`
- [ ] Phase 3 — Deep models + generalization (4–5 days) → `v0.3`
- [ ] Phase 4 — Conformal intervals + explainability (3 days) → `v0.4`
- [ ] Phase 5 — Maintenance policy simulator (2 days) → `v0.5`
- [ ] Phase 6 — ONNX, API, dashboard, deploy (3–4 days) → `v0.9`
- [ ] Phase 7 — README, report, polish (2 days) → `v1.0`

---

## Phase 0 — Scaffold

```text
Read CLAUDE.md and docs/SPEC.md sections 4, 11 and 12. Scaffold the repository exactly as in SPEC §4:
- uv project (Python 3.11) with deps: torch, numpy, pandas, pyarrow, h5py, scikit-learn, lightgbm, hydra-core, mlflow, captum, onnx, onnxruntime, fastapi, uvicorn, pydantic, streamlit, plotly, matplotlib; dev deps: pytest, ruff, mypy, pandas-stubs.
- torch comes from the PyTorch CPU index (SPEC §12): a [[tool.uv.index]] named "pytorch-cpu" at https://download.pytorch.org/whl/cpu with explicit = true, plus [tool.uv.sources] torch = { index = "pytorch-cpu" }. Confirm no nvidia-* packages land in uv.lock.
- ruff (line length 100, rules E,F,I,B,UP,SIM), mypy (strict on src/aerosentinel, with a [[tool.mypy.overrides]] block setting ignore_missing_imports = true for h5py, lightgbm, captum, mlflow, onnxruntime), pytest with a "slow" marker, all configured in pyproject.toml.
- Makefile targets: setup, data-cmapss, data-ncmapss (both call scripts/download_data.py; stub until Phases 1–2), test, test-all, lint, train, report (calls scripts/make_tables.py once it exists), app, api.
- .gitignore covering data/, mlruns/, outputs/, *.ckpt, *.onnx (except app/demo_data). Create a tracked configs/data_checksums.txt with a header line only.
- GitHub Actions CI: uv sync, ruff check, mypy, pytest -m "not slow" on push and PR.
- src/aerosentinel/utils.py with seed_everything and a paths module.
- A placeholder README with title and the one-line pitch from SPEC §0.
- tests/test_smoke.py that imports the package.
Verify: run make lint and make test and show the output. Git is already initialized; make the first commit "chore: scaffold project".
```

---

## Phase 1 — C-MAPSS baselines → v0.1

```text
Implement SPEC §1.1, §1.3 (items 2, 5, 6 for C-MAPSS), §2 and §5.1.
1. scripts/download_data.py cmapss → aerosentinel.data.download:
   - Download C-MAPSS from the primary URL only (the data.nasa.gov mirror returns 403; no fallback).
   - Write to data/raw/; this script is the only writer there.
   - Record SHA256 + size in the tracked configs/data_checksums.txt.
   - Locate the train/test/RUL files inside the nested CMAPSSData.zip.
2. data/cmapss.py:
   - Parse the 26 columns and add RUL (piecewise cap 125, configurable).
   - Regime normalization: KMeans k=6 on settings for FD002/FD004, fit on train only.
   - THEN the variance-threshold sensor drop, fit on train after regime normalization.
3. data/windows.py and data/splits.py: unit-level splits (val = 20% of train units), windows of length 30 with stride 1, last-window extraction for test units, padding for short test units (document the choice).
4. metrics.py with RMSE, MAE, Saxena score (sum and mean) and PHM21 combined score.
5. Models: LightGBM on engineered window features, 1D-CNN, LSTM. Hydra configs + MLflow logging. 3 seeds.
6. evaluate.py writes reports/results/<dataset>_<model>_<setting>.json (SPEC §4 naming, e.g. cmapss-fd001_lgbm_cap125.json), with metrics against capped (primary) and uncapped test truth.
7. scripts/make_tables.py builds every markdown table from the results JSONs and writes them into README between marker comments. Every later phase reuses and extends this one script.
8. Tests from SPEC §11: splits, windows, labels, metrics, no-leak features.
Targets (capped test truth): FD001 RMSE ≤ 13, FD004 ≤ 20. If any capped FD001 result is below 11, stop and hunt for leakage before continuing.
Verify: full table for FD001–FD004 × 3 models (mean ± std, capped and uncapped), written into README under "Results at a glance" by scripts/make_tables.py; test output.
```

Review prompt:
```text
Use the ml-reviewer subagent to review the Phase 1 diff against docs/SPEC.md §1.3, §2 and §11. Report only correctness, leakage and protocol issues.
```

---

## Phase 2 — N-CMAPSS pipeline + EDA → v0.2

```text
Implement SPEC §1.2 and §1.3 for N-CMAPSS. Model DS02 first; extract DS01–DS07 in the same download because Phase 4 pools them.
1. scripts/download_data.py ncmapss, per SPEC §1.2:
   - curl -sI to confirm Content-Length (expect ~15.76 GB).
   - Check free disk ≥ ~50 GB.
   - Download the whole archive (resumable).
   - Stream-extract only the DS01–DS07 .h5 files from the nested data_set.zip, without writing data_set.zip to disk.
   - Record SHA256 + size of the zip and each .h5 in configs/data_checksums.txt.
   - Delete the zip.
2. data/ncmapss.py:
   - List HDF5 keys and print shapes/var names before writing any loader logic.
   - Build a chunked h5py reader that writes per-unit contiguous float32 arrays (.npy) with W + X_s + Y + A. A and Y are kept as labels/metadata, never features.
   - Handle dev and test splits.
3. Flight-class audit (SPEC §5.5): a script prints Fc per DS02 and DS03 dev unit and writes reports/results/ncmapss-ds02_data_fc-audit.json, recording which dataset leave-one-class-out will use. DS02 if every class has ≥ 2 dev units, else DS03.
4. Cap check (SPEC §1.3 item 2):
   - Compare the η=5% caps (and the cruise mask) against the CruiseBench repo (github.com/NostalgiaJohn/CruiseBench); record the commit you checked.
   - Cross-check the caps against the minimum abnormal-degradation length from hs on dev units.
   - If anything disagrees, stop and report before using the caps.
5. data/masks.py: cruise-stage mask per SPEC §1.3 item 4, with tests on a synthetic altitude profile. Windows must lie inside the contiguous cruise run.
6. data/baseline_residuals.py, per SPEC §1.3 item 5:
   - Per-engine MLP W → X_s fit on that engine's own first 5 cycles (no hs, no other engine); add residual channels.
   - Log the W-extrapolation share per engine.
   - Test that engine u's baseline sees only u's cycles 1–5.
7. data/windows.py: a window index (unit, cycle, start) over the per-unit arrays, keyed by config hash, with windows sliced on the fly and no materialized window cache.
   - T=256 with configurable stride and downsample_factor; test stride = train stride.
   - Windows never cross a (unit, cycle) boundary or a cruise-run gap.
   - Per-cycle prediction = median of the cycle's window predictions.
   - Report the fraction of cycles with ≥ 1 window per (setting, downsample) and propose T for downsample 10 (SPEC §1.3 item 3 open point).
8. Dataset-wise RUL caps from SPEC §1.3 item 2 (after step 4).
9. notebooks/02_eda_ncmapss.ipynb + scripts/make_eda_figures.py, saved to reports/figures/:
   - flight profiles by class
   - sensor trajectories over life for one unit
   - hs onset vs cycle
   - healthy-baseline residuals over life (hypothesis: they drift with degradation)
   - train vs test units 14/15 altitude distribution
10. LightGBM baseline on DS02 at downsample 10 (windows → engineered features) as the first N-CMAPSS number. Per-cycle metrics are primary, per-window secondary.
Verify: download log with checksums, Fc audit JSON, cap-check result, cycles-with-window table, figures saved, LightGBM DS02 results JSONs (full_flight and cruise_only), all tests green. Update README with an "N-CMAPSS" section and 2 figures.
```

---

## Phase 3 — Deep models + generalization → v0.3

```text
Implement SPEC §5.2, §5.3, §5.4, §5.5 on DS02: the full grid at downsample 10 on CPU, then full resolution for the top 2 models on Kaggle (SPEC §12).
1. Models: TCN, TSMixer, Patch Transformer, each < 1M params, shared training loop with AdamW, cosine schedule, early stopping on val per-cycle RMSE, grad clipping, AMP on GPU.
2. Multi-task heads (RUL + hs + theta) behind a config flag, for the neural models only. Inputs remain W + X_s (+ residuals) only. Add a test that asserts the model forward pass receives exactly the allowed channels.
3. Grid at downsample 10 on CPU (Hydra experiment configs named per SPEC §4, 3 seeds each), using the T fixed in Phase 2:
   - LightGBM single-task × {full_flight, cruise_only}
   - {TCN, TSMixer, Transformer} × {single-task, multi-task} × {full_flight, cruise_only}
4. Rank models by DS02 validation per-cycle RMSE (never test) and freeze the ranking. Rerun the top 2 at full resolution on Kaggle via scripts/kaggle_train.sh. The script installs the package from GitHub and keeps Kaggle's preinstalled CUDA torch.
5. Save OOF predictions on all dev units from the unit rotation (SPEC §1.3 item 6) for Phases 4–5.
6. Generalization:
   - Leave-one-flight-class-out on the dataset chosen by the Phase 2 Fc audit (DS02 or DS03).
   - Official DS02 test units 11, 14, 15 reported per unit.
7. Extend scripts/make_tables.py, plus figures: RUL trajectories per test unit, model comparison bars, per-flight-class error.
Verify: results JSONs for all runs, tables in README, and a short "Findings" paragraph that checks each stated hypothesis and lists what helped and what didn't, including negative results.
```

Review prompt:
```text
Use the ml-reviewer subagent to review the Phase 3 diff. Focus on: any banned feature reaching the model, split leakage, cap/target construction, windows crossing (unit, cycle) boundaries, test stride ≠ train stride, whether the top-2 selection used validation only, whether saved OOF predictions are truly out-of-fold, and whether comparisons used identical budgets and seeds.
```

---

## Phase 4 — Conformal intervals + explainability → v0.4

```text
Implement SPEC §6 and §8.
1. Quantile head with pinball loss (q_lo, q_med, q_hi):
   - on the best Phase 1 deep model for C-MAPSS FD002/FD004
   - on the best Phase 3 model for N-CMAPSS (pooled DS01–DS07 at downsample 10, and DS02)
2. uncertainty/conformal.py, asymmetric CQR at total alpha = 0.10:
   - Alpha rule: n_cal ≥ 60 → (alpha_lo, alpha_hi) = (0.02, 0.08); otherwise (0.05, 0.05).
   - cqr_unit = unit-cross-fitted (K=5, CV+ style), one random cycle per calibration unit, R=20 draws.
   - cqr_split = window-level.
   - Unit tests: the slow synthetic coverage test, the alpha-rule test, and the one-cycle-per-unit test.
3. Runs:
   - cqr_unit and cqr_split on C-MAPSS FD002, FD004 and pooled N-CMAPSS DS01–DS07.
   - DS02: window-level CQR only, with the SPEC §6 caveat on every DS02 table and figure.
4. Baselines: MC dropout and a 5-member deep ensemble, uncalibrated.
5. Report per method: marginal per-cycle coverage (mean ± std over draws), lower-tail violation rate, mean width, coverage by RUL bucket, plus the alpha pair and n_cal used.
6. explain/attributions.py: integrated gradients on the residual channels of the with-residuals model, aggregated per unit; map sensors → components; SHAP for LightGBM. Figures to reports/figures/.
Verify:
- results JSONs, and a table comparing cqr_unit vs cqr_split vs MC dropout vs ensemble on FD002, FD004 and pooled N-CMAPSS
- DS02 interval trajectory plots for units 11, 14, 15, labelled illustrative
- an attribution figure with a one-paragraph interpretation stating whether the hypothesis (turbine-section residuals dominate for DS02) held
```

---

## Phase 5 — Maintenance policy simulator → v0.5

```text
Implement SPEC §7 in decision/policy.py.
1. Walk-forward simulator per test unit, per SPEC §7:
   - Lead time L: a decision at t takes effect at t+L.
   - Policies: fixed interval, point threshold, conformal lower-bound threshold, and oracle (decides at t_fail − 2L, removes at t_fail − L).
   - Residual-channel models start at cycle 6.
   - Headline on pooled N-CMAPSS test units; DS02 shown with the window-level caveat.
2. Tuning: T* on dev-unit lifetimes; every threshold/margin on OUT-OF-FOLD dev-unit predictions only (saved in Phase 3), never in-sample predictions and never test units.
3. Configurable costs (C_p, C_f, waste per cycle). Sensitivity runs for C_f/C_p in {5, 10, 50} and alpha in {0.05, 0.1, 0.2}. Headline at alpha = 0.10, with the alpha split per the SPEC §6 rule.
4. Tests:
   - The oracle has zero failures and waste of exactly L.
   - A never-remove policy fails every engine.
   - Tuning functions reject non-OOF predictions.
5. Cost table via scripts/make_tables.py; scripts/make_policy_report.py draws the cost-vs-alpha curve and the failure-rate-vs-waste scatter.
Verify: results JSON, figures, and a README section "From predictions to decisions" stating cost assumptions explicitly.
```

---

## Phase 6 — ONNX, API, dashboard, deploy → v0.9

```text
Implement SPEC §9. Load the frontend-design skill if available before building the dashboard UI.
1. serve/export_onnx.py + parity test (max abs diff < 1e-4) + latency benchmark script (p50/p95, batch 1, CPU).
2. serve/api.py: FastAPI /health and /predict with pydantic validation (shape [T,18], no NaNs). If the served model uses residual channels, /predict also takes engine_id and loads that engine's ONNX baseline MLP; unknown engines return 404 (SPEC §9). Tests with TestClient.
3. Dockerfile for the API using onnxruntime only (no torch). Verify the image builds and /health responds. Needs Docker Desktop's WSL integration enabled.
4. scripts/build_demo_bundle.py: precompute predictions, intervals, attributions and flight profiles for test units. Commit to app/demo_data/ if ≤ 10 MB; otherwise publish as a Hugging Face dataset that the app downloads at start.
5. app/dashboard.py "Fleet Control Room" exactly as SPEC §9 describes, reading only the demo bundle.
6. First check the current Hugging Face Spaces SDK options (is Streamlit a native SDK, or is the Docker SDK required?) and report what you found. Then add the matching deploy files (app entrypoint + requirements or Dockerfile) and instructions in README.
Verify: screenshots of each dashboard view, Docker build log, API test output. Record a 15–20 s GIF of the dashboard (I'll record it if you can't) and add it to the README top.
```

---

## Phase 7 — README, report, polish → v1.0

```text
Implement SPEC §10 and §13.
1. Rewrite README.md in the exact section order of SPEC §10. All numbers pulled from reports/results JSONs via scripts/make_tables.py — no hand-typed metrics.
2. Mermaid architecture diagram in README.
3. reports/TECHNICAL_REPORT.md (4–6 pages): problem, data, protocol, models, uncertainty, decision layer, results, limitations, references.
4. MODEL_CARD.md: intended use, data, metrics, limitations, ethical note (simulated data, not for real airworthiness decisions).
5. Fresh-clone test: clone into a temp dir, run make setup && make test && make app (smoke), fix anything that breaks.
6. Fill the resume template in SPEC §13 with real numbers and save it to reports/RESUME_BULLETS.md.
Verify: fresh-clone log, rendered README check (headings, badges, GIF, tables), and the filled resume bullets.
```

Final review prompt:
```text
Use the ml-reviewer subagent to audit the whole repo as a skeptical hiring manager and ML interviewer would: claims not backed by a results JSON, leakage risks, broken quickstart steps, confusing README sections. Report the top 10 issues by severity.
```
