# AeroSentinel — Full Project Specification

> Uncertainty-aware remaining-useful-life (RUL) prediction and maintenance decision-making for turbofan engines, on NASA's N-CMAPSS dataset (simulated degradation under **real recorded flight profiles**), benchmarked first on the classic C-MAPSS.

This document is the single source of truth. Claude Code: read the section relevant to the current phase in `docs/PHASES.md` before implementing; don't re-derive decisions recorded here.

---

## 0. Why this project (and what makes it stand out)

The Instagram version of this project — "train an LSTM on C-MAPSS to predict cycles left" — exists in thousands of GitHub repos and Kaggle notebooks. A reviewer has seen it. What makes AeroSentinel different is the layer **around** the model, which is what real ML engineering looks like:

| Typical repo | AeroSentinel |
|---|---|
| C-MAPSS FD001 only (one operating condition) | C-MAPSS FD001–FD004 as warm-up, then **N-CMAPSS** with full flight profiles (climb/cruise/descent) |
| One point prediction | **Calibrated prediction intervals** (conformalized quantile regression, asymmetric — late predictions are the dangerous ones) |
| Random train/test split (often leaky) | Split by engine unit; leakage tests in CI; **cross-flight-class generalization** experiment |
| "RMSE = 12.3" | RMSE + NASA asymmetric score + interval coverage/width + **maintenance cost saved vs fixed-interval policy** |
| Notebook | Package with tests, CI, Hydra configs, MLflow tracking, ONNX export, FastAPI service, Docker, live Streamlit demo |
| Black box | Integrated-gradients sensor attributions mapped to engine components |

The headline story for interviews: *"Point predictions are useless to a maintenance planner. I built a system that says 'this engine has 18–31 flights left at 90% coverage, plan a shop visit within the next N flights', measured whether that interval really covers 90% on held-out engines in the settings with enough engines to calibrate on (C-MAPSS FD002/FD004, pooled N-CMAPSS DS01–DS07), and tested whether a policy built on it cuts simulated maintenance cost vs. a fixed schedule."* (The numbers in this sentence are placeholders; the real ones come only from results JSONs.)

**Interval target everywhere: 90% (total α = 0.10).** Other α values appear only in the §7 sensitivity sweep. Expected outcomes throughout this spec are **hypotheses**, to be reported whichever way they turn out.

---

## 1. Datasets

### 1.1 C-MAPSS (warm-up benchmark — Phase 1)
- **Source:** NASA PCoE Data Repository, item 6 "Turbofan Engine Degradation Simulation".
  Download (primary URL only): `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip` (12.4 MB, verified 2026-11-27).
  No fallback mirror: `https://data.nasa.gov/docs/legacy/CMAPSSData.zip` returned 403 on 2026-11-27.
- **Citation:** A. Saxena and K. Goebel (2008). "Turbofan Engine Degradation Simulation Data Set", NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA.
- **Format:** space-separated text, 26 columns: `unit, cycle, setting1..3, s1..s21` (the readme says "sensor 26" but there are 21 sensors). Files: `train_FD00x.txt`, `test_FD00x.txt`, `RUL_FD00x.txt`.
- **Subsets:**

| Subset | Train units | Test units | Operating conditions | Fault modes |
|---|---|---|---|---|
| FD001 | 100 | 100 | 1 (sea level) | 1 (HPC) |
| FD002 | 260 | 259 | 6 | 1 (HPC) |
| FD003 | 100 | 100 | 1 | 2 (HPC, Fan) |
| FD004 | 249 | 248 | 6 | 2 (HPC, Fan) |

- Train trajectories run to failure. Test trajectories stop before failure; `RUL_FD00x.txt` gives the true RUL at the last test cycle. Standard evaluation uses **only the last window of each test unit**.
- Test metrics are reported against two versions of the truth: **capped at 125 (primary)** and uncapped (secondary). The leakage alarm (§3) applies to the capped number.
- Note: the primary archive wraps a nested `CMAPSSData.zip`. The download script must find `train_FD001.txt` wherever it is.

### 1.2 N-CMAPSS (main dataset — Phases 2+)
- **Source:** NASA PCoE Data Repository, item 17 "Turbofan Engine Degradation Simulation-2".
  Download: `https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip`
- **Citation:** M. Arias Chao, C. Kulkarni, K. Goebel, O. Fink (2021). "Aircraft Engine Run-to-Failure Dataset under Real Flight Conditions for Prognostics and Diagnostics." *Data* 6(1):5. doi:10.3390/data6010005.
- **What it is:** CMAPSS simulation driven by real flight data recorded on a commercial jet; complete flights (climb, cruise, descent) at native time resolution, not one snapshot per cycle. Engines start with unknown, different initial health; degradation is operation-history-dependent with initial, normal and abnormal stages; the dataset includes between-flight maintenance effects within limits.
- **Archive (verified 2026-11-27):** a single **15.76 GB** zip that wraps one deflate-compressed inner `data_set.zip` (15.81 GB). There is no per-subset download, and the inner zip can't be range-read. Procedure (all inside `scripts/download_data.py ncmapss`):
  1. `curl -sI` to confirm Content-Length.
  2. Check free disk ≥ ~50 GB: the 15.8 GB zip plus the `.h5` files. DS01–DS07 are an estimated 20–30 GB, unmeasured.
  3. Download the whole archive once (resumable).
  4. **Stream-extract only the needed `.h5` files** (DS01–DS07: DS02 for modelling, the rest for pooled conformal, §6) through the nested zip without writing `data_set.zip` to disk. For example, Python `zipfile` on the nested member, extracting members in offset order, takes ~2 sequential decompression passes and no temp file.
  5. Record SHA256 and size of the zip and each `.h5` in `configs/data_checksums.txt`.
  6. **Delete the zip afterwards.**
- **Files (HDF5):** `N-CMAPSS_DS01-005.h5`, `DS02-006`, `DS03-012`, `DS04`, `DS05`, `DS06`, `DS07`, `DS08a-009`, `DS08c-008`, `DS08d-010`. A 2026 paper (CruiseBench) reports DS08d-010 could not be read — skip it if h5py fails.
- **Expected HDF5 keys** (verify by listing keys first; don't assume):
  `W_dev, X_s_dev, X_v_dev, T_dev, Y_dev, A_dev` and the same with `_test`, plus name arrays `W_var, X_s_var, X_v_var, T_var, A_var`.
- **Variable groups:**
  - `W` — scenario descriptors (4): `alt` [ft], `Mach` [-], `TRA` throttle-resolver angle [%], `T2` fan inlet total temperature [°R]
  - `X_s` — measured sensors (14): `T24, T30, T48, T50, P15, P2, P21, P24, Ps30, P40, P50, Nf, Nc, Wf`
  - `X_v` — virtual (unmeasurable) sensors — **not an input**
  - `T` / theta — health parameters (component efficiency/flow modifiers) — **not an input**; usable as an auxiliary *training target* (see §5.4)
  - `Y` — RUL label in flight cycles
  - `A` — auxiliary: `unit`, `cycle`, `Fc` (flight class), `hs` (health state: 1 healthy, 0 faulty) — **not inputs**; `hs` is usable as an auxiliary target
- **Flight classes:** 1 = 1–3 h flights, 2 = 3–5 h, 3 = > 5 h.
- **DS02 (start here):** 9 units with flight classes 1–3. Dev (train) units 2, 5, 10, 16, 18, 20; test units 11, 14, 15. Test units 14 and 15 fly shorter, lower-altitude flights than the training units — a built-in distribution shift. Failure modes: HPT efficiency degradation plus LPT efficiency & flow. The flight class of each dev unit is **not** verified here; Phase 2 reads it from `A_dev` (see §5.5).
- **Scale:** each subset holds millions of rows. Typical engine life is ~60–100 flight cycles. Plan memory accordingly (float32, chunked reads, downsampling).

### 1.3 Data-handling decisions (fixed)
1. Inputs = `W + X_s` (18 channels). This is the original data-driven setting from Arias Chao et al. and the CruiseBench protocol.
2. **RUL target capping:**
   - C-MAPSS: piecewise linear, cap = 125 cycles (community standard) on training targets. Test truth is reported **capped at 125 (primary)** and uncapped (secondary).
   - N-CMAPSS: dataset-wise caps derived from the minimum abnormal-degradation length. Default to CruiseBench's η = 5% caps (DS01: 38, DS02: 34, DS03: 34, DS04: 43, DS05: 38, DS06: 41, DS07: 39, DS08a: 34, DS08c: 28) and also report uncapped. These caps are **unverified**: Phase 2 checks them against the CruiseBench repo, and cross-checks them against the minimum abnormal-degradation length computed from `hs` on dev units, before using them. Cap is part of the experiment name.
3. **Temporal reduction and windowing for N-CMAPSS:** native-resolution windows (default T = 256 samples) with a stride (default 10), NOT naive row downsampling. Rationale: CruiseBench ablation showed stride preserves within-window dynamics and gave lower error on most subsets. **Test stride = train stride** (one config value). Keep a `downsample_factor` config for laptops; default 1 on GPU, 10 on CPU.
   - Windows **never cross a (unit, cycle) boundary**: each flight is windowed on its own. A flight with fewer than T samples in the active setting contributes no window.
   - **Per-cycle prediction** = median of that cycle's window predictions. For quantile models, take the median of each quantile across the cycle's windows. **Primary metrics are per cycle**; per-window metrics are secondary. Cycles with no valid window are excluded, and their count is reported per setting.
   - Windows are never materialized. Store per-unit contiguous arrays plus a window index (§12).
   - *Open, resolved in Phase 2:* at downsample 10, T = 256 samples spans ~43 min of flight, so many short (class 1) cruise runs may yield no window. Phase 2 reports the fraction of cycles with ≥ 1 window per (setting, downsample) and fixes T for downsample 10 before Phase 3.
4. **Flight-stage handling:** two settings, both reported separately:
   - `full_flight`: all samples.
   - `cruise_only`: altitude-plateau mask (rolling-median altitude, find longest stable high-altitude run ≥ 512 samples, std < 50 ft). Windows lie entirely inside that contiguous run; non-contiguous cruise samples are **never spliced** into one window. Implement our own mask function, test it, and compare with CruiseBench's released mask (github.com/NostalgiaJohn/CruiseBench).
5. **Normalization:** fit scalers on training units only. For C-MAPSS FD002/FD004, per-operating-regime normalization (KMeans k=6 on the 3 settings, fit on train). The near-constant-sensor variance threshold runs **after** regime normalization.
   - For N-CMAPSS, standardize X_s **conditioned on W** with a **per-engine healthy baseline**. For each engine (train, val, calib and test alike), fit a small MLP W → X_s on **that engine's own first 5 cycles**, using no `hs` and no other engine, and feed residuals `X_s − f_u(W)` as extra channels.
   - The baseline uses only the engine's own past, so it is legal at inference, and it absorbs each engine's unknown initial health.
   - Residuals on cycles 1–5 are in-sample (near zero by construction). Models with residual channels therefore predict from cycle 6 onward in walk-forward use (§7).
   - Per engine, log the share of later samples whose W falls outside the range seen in cycles 1–5; this is the extrapolation risk.
6. **Splits:** by unit only. For N-CMAPSS dev units, carve out validation and calibration units by leave-unit-out (e.g., DS02: 4 train / 1 val / 1 calib, rotated over folds). Official test units are touched only by the final evaluation script.
   - The rotation holds out every dev unit exactly once, giving **out-of-fold (OOF) predictions** for all dev units.
   - Everything tuned on dev units uses OOF predictions only, never in-sample ones: policy thresholds, the margin, and unit-level conformal calibration.

---

## 2. Metrics

All metrics live in `src/aerosentinel/metrics.py` with unit tests against hand-computed values.

- **RMSE** (cycles), **MAE**.
  - N-CMAPSS: computed **per cycle** (primary, §1.3 item 3) and per window (secondary).
  - C-MAPSS: one window corresponds to one cycle, and test metrics use the last cycle of each test unit against capped (primary) and uncapped truth.
- **NASA / Saxena asymmetric score:** with d = ŷ − y (prediction minus truth), s = Σ (e^{−d/13} − 1) if d < 0 (early), Σ (e^{d/10} − 1) if d ≥ 0 (late). Late predictions are punished harder. Report the **sum** for C-MAPSS (to match literature) and the **mean** for N-CMAPSS (PHM 2021 challenge convention). Also implement the PHM21 combined score `0.5·RMSE + 0.5·mean_score`.
- **Interval metrics:** empirical **marginal per-cycle coverage** (the target, §6), mean width, coverage by RUL bucket (0–10, 10–30, 30+), and *lower-tail violation rate* P(y < lower bound) — the safety-critical number.
- **Decision metrics (§7):** expected cost per engine, unscheduled failure rate, mean wasted life (cycles removed early).
- **Efficiency:** parameter count, ONNX CPU latency (batch 1, p50/p95), model size MB.

---

## 3. Reference numbers (for sanity, NOT for SOTA claims)

C-MAPSS RMSE reported in the literature (protocols differ, treat as ballpark):
- Bi-LSTM: 17.60 / 29.67 / 17.62 / 31.84 (FD001–4)
- DA-CNN: 11.78 / 16.95 / 11.56 / 18.23
- Transformer (single-condition): 11.36 FD001, 11.28 FD003
- Strong 2025 hybrids: ~11.0 FD001, ~13.3 FD002, ~11.1 FD003, ~18.3 FD004

**Target for Phase 1 (capped test truth):** FD001 RMSE ≤ 13, FD004 ≤ 20 with a clean protocol. If capped FD001 comes in under ~11, suspect leakage before celebrating. The literature numbers above mix capped and uncapped test truth, which is why we report both.

N-CMAPSS: numbers across papers are **not comparable** — they differ in subsets, downsampling (×1 to ×1000), window size, caps, and feature groups (CruiseBench Table 2.1). Under CruiseBench's cruise-only η=5% protocol, simple baselines averaged RMSE ≈ 3.5 (TSMixer) to 3.9 (TCN) across 9 subsets; LSTM was much worse and unstable. PHM 2021 challenge winner used a deep CNN handling variable-length input. We report our own protocol clearly and re-run baselines under it.

---

## 4. Repository layout

```
aerosentinel/
├── CLAUDE.md
├── README.md                     # the showcase — see §10
├── pyproject.toml                # uv-managed; ruff, mypy, pytest config here
├── uv.lock
├── Makefile                      # make setup / data / test / train / report / app
├── Dockerfile                    # API image
├── .github/workflows/ci.yml      # ruff, mypy, pytest (not slow), on push/PR
├── configs/                      # Hydra
│   ├── config.yaml
│   ├── data/{cmapss_fd001..4,ncmapss_ds02,...}.yaml
│   ├── model/{lgbm,cnn1d,lstm,tcn,transformer,tsmixer}.yaml
│   ├── experiment/*.yaml         # named <dataset>_<model>_<setting> (see below)
│   └── data_checksums.txt        # TRACKED: SHA256 + size of every downloaded archive/.h5
├── src/aerosentinel/
│   ├── data/
│   │   ├── download.py           # resumable downloads, nested-zip stream extraction, checksums
│   │   ├── cmapss.py             # load, label RUL, regime normalization
│   │   ├── ncmapss.py            # chunked h5py reader → per-unit parquet / .npy arrays
│   │   ├── windows.py            # window INDEX over per-unit arrays; never crosses (unit, cycle)
│   │   ├── masks.py              # cruise-stage mask
│   │   ├── baseline_residuals.py # per-engine W→healthy X_s MLP (own first 5 cycles)
│   │   └── splits.py             # unit-level folds
│   ├── models/
│   │   ├── lgbm.py               # feature-engineered GBM baseline
│   │   ├── cnn1d.py, lstm.py, tcn.py, tsmixer.py, transformer.py
│   │   └── heads.py              # multi-task heads: RUL, quantiles, hs, theta
│   ├── train.py                  # Hydra entrypoint; MLflow logging
│   ├── evaluate.py               # writes reports/results/<dataset>_<model>_<setting>.json
│   ├── metrics.py
│   ├── uncertainty/conformal.py  # asymmetric CQR: window-level + unit-level (1 random cycle/unit)
│   ├── explain/attributions.py   # integrated gradients via captum
│   ├── decision/policy.py        # maintenance policies + fleet simulator
│   ├── serve/{export_onnx.py, api.py, schemas.py}
│   └── utils.py                  # seeding, logging, paths
├── app/dashboard.py              # Streamlit "Fleet Control Room"
├── scripts/                      # one script per reported table/figure, plus:
│   ├── download_data.py          #   the ONLY writer of data/raw/ (cmapss | ncmapss)
│   └── make_tables.py            #   all markdown tables from results JSONs → README markers (from Phase 1)
├── tests/                        # pytest; mark heavy ones @pytest.mark.slow
├── notebooks/01_eda_cmapss.ipynb, 02_eda_ncmapss.ipynb   # exploration only
├── reports/
│   ├── results/*.json            # machine-written metrics, <dataset>_<model>_<setting>.json
│   ├── figures/*.png
│   └── TECHNICAL_REPORT.md       # 4–6 page write-up
├── MODEL_CARD.md
└── data/{raw,processed}/         # gitignored; raw written only by scripts/download_data.py
```

**Results naming (single scheme):** `reports/results/<dataset>_<model>_<setting>.json`. Rules:
- Lowercase. `_` separates the three fields; `-` joins words inside a field.
- `dataset` ∈ `cmapss-fd001…cmapss-fd004`, `ncmapss-ds01…ncmapss-ds07`, `ncmapss-pooled`.
- `model` ∈ `constant`, `ridge` (naive baselines), `lgbm`, `cnn1d`, `lstm`, `tcn`, `tsmixer`, `transformer`, or `data` for data audits.
- `setting` encodes everything else that defines the run, e.g. `single-cruise-cap34-ds10`, `cqr-unit-a10`, `policy-cf10`.
- The Hydra experiment name equals the file stem, so `configs/experiment/<stem>.yaml` → `reports/results/<stem>.json`.

---

## 5. Modeling

### 5.1 Baselines (Phase 1, C-MAPSS; re-run on N-CMAPSS in Phase 3)
1. **LightGBM on engineered features:** per window — last value, mean, std, slope (linear fit), min/max of each sensor + regime id. Fast, strong, explainable. Many papers underrate this.
2. **1D-CNN** (Li et al. 2018 style): window 30, 5 conv layers, dropout.
3. **LSTM** (2 layers, 64 hidden).
Drop near-constant sensors on C-MAPSS (variance threshold fit on train, applied **after** regime normalization for FD002/FD004; FD001 typically s1, s5, s6, s10, s16, s18, s19).

### 5.2 Main architectures (Phase 3)
- **TCN** (dilated causal convs, residual blocks) — strong and stable per CruiseBench.
- **TSMixer** — best average in CruiseBench cruise-only; less stable on full flights.
- **Patch Transformer** (PatchTST-style: patch length 16, stride 8, channel-independent embedding, 3 encoder layers) — the "modern" model; good for attention plots.
Keep each under ~1M params. Same training budget for all.

### 5.3 Training defaults
AdamW, lr 3e-4 with cosine decay + warmup, batch 256–512, early stopping on val RMSE (patience 10), gradient clipping 1.0, mixed precision on GPU, 3 seeds. Loss = MSE for point models; see §6 for quantile training.

### 5.4 Multi-task variant (the differentiator inside the model)
Neural models only (TCN, TSMixer, Transformer); there is no LightGBM multi-task variant. Shared encoder, three heads:
- RUL (regression, capped target),
- health state `hs` (binary — "has abnormal degradation started?"),
- health parameters theta at window end (regression, auxiliary).
Loss = L_rul + λ1·BCE(hs) + λ2·MSE(theta), λ tuned on val. These labels are **targets only during training**, never inputs — at inference the model sees W + X_s (+ residuals) only. Hypothesis: auxiliary supervision improves RUL and robustness to unseen flight classes. Report either way; a negative result is still a result.

### 5.5 Generalization experiments
1. **Leave-one-flight-class-out:** train on two classes, test on the third. Phase 2 first reads `Fc` per dev unit from `A_dev` for DS02, and for DS03 likewise.
   - If every flight class has ≥ 2 DS02 dev units, run it on DS02.
   - If any class has < 2, run it on **DS03 instead**. DS03 is reported to have equal units per class; verify that from `A_dev` the same way.
2. **Official DS02 test units** 11 (in-distribution profile), 14 and 15 (shorter, lower flights) — report per unit.
3. Optional stretch: train on DS01–DS03, test on DS04+ (new failure modes). Hypothesis: error rises on unseen failure modes; analyze either way.

---

## 6. Uncertainty — conformalized quantile regression (Phase 4)

- Train the model with a quantile head (pinball loss) for q_lo, q_med, q_hi. On N-CMAPSS, per-cycle quantiles = median of the cycle's window quantiles (§1.3 item 3).
- **Coverage target: 90% marginal per-cycle coverage (total α = 0.10).** For a new engine and a random cycle of its life, P(true RUL ∈ interval) ≥ 0.90.
- **Asymmetric split, decided by the number of calibration units n_cal:**
  - If n_cal ≥ 60: α_lo = 0.02 and α_hi = 0.08. The lower bound triggers maintenance, and a true RUL below it means the engine fails earlier than planned. This follows the 2026 IJPHM risk-aware CQR paper.
  - Otherwise: α_lo = α_hi = 0.05.
  - Why: a split-conformal bound at miscoverage α is finite only if n_cal ≥ ⌈(1−α)/α⌉, which is 49 units at α = 0.02; 60 leaves margin.
  - The lower and upper tails are calibrated separately. The rule is applied in code, and the α pair and n_cal are written into every results JSON.
- **Unit-level calibration (`cqr_unit`): the headline method, run only where units are plentiful.** Calibrate on **one random cycle per calibration unit**, which makes scores exchangeable across engines. Repeat the random-cycle draw R = 20 times and report coverage mean ± std over draws.
  - **C-MAPSS FD002 and FD004** (260 / 248 train units, so n_cal ≥ 60 and α = 0.02 / 0.08):
    - Unit-cross-fit with K = 5 folds over train units. Each fold model scores its held-out units, giving OOF nonconformity scores for every train unit.
    - Test intervals follow CV+ (Barber et al. 2021) adapted to CQR: each calibration unit's score is combined with the prediction of the fold model that never saw that unit. The worst-case guarantee is 1 − 2α, so we report empirical coverage.
    - Coverage is measured on the official test units (last cycle each; capped and uncapped truth).
  - **Pooled N-CMAPSS DS01–DS07 dev units:**
    - Same unit-cross-fitting over all pooled dev units: one model, with dataset-wise caps on targets.
    - Coverage is measured on all cycles of the pooled official test units.
    - n_cal is expected to fall below 60, giving α = 0.05 / 0.05, but the code decides from the actual count.
    - Runs at the Phase 3 downsample-10 setting.
- **`cqr_split` (window-level, naive comparison):** same models and data, but calibration scores pool all calibration windows. Hypothesis: windows within an engine are correlated, so this under-covers on new engines. The gap is a talking point only if it actually appears.
- **DS02 (6 dev units): window-level CQR only, illustrative.**
  - Calibrate on the windows of the held-out calib unit(s) in the rotation, with α_lo = α_hi = 0.05 (n_cal < 60).
  - Every DS02 interval table and figure carries this caveat verbatim: *"Window-level calibration on at most 6 engines. Windows from one engine are not exchangeable, so no coverage guarantee holds — illustrative only."*
  - The DS02 trajectory plots (units 11, 14, 15) use these intervals.
- Uncalibrated comparisons: MC dropout and a 5-member deep ensemble. Hypothesis: they under-cover. Report whatever happens.
- Figure: RUL trajectory per test unit with true RUL, median prediction and shaded interval.

References: Javanmardi & Hüllermeier, "Conformal Prediction Intervals for Remaining Useful Lifetime Estimation", IJPHM 2023 (doi:10.36001/ijphm.2023.v14i2.3417); Romano et al. 2019 (CQR); Barber, Candès, Ramdas, Tibshirani 2021, "Predictive inference with the jackknife+" (CV+); IJPHM 2026, "Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression".

---

## 7. Decision layer — maintenance policy simulator (Phase 5)

Turn predictions into decisions and money.

- **Simulator:** for each test unit, walk forward cycle by cycle. At each cycle the policy sees only data up to that cycle and decides `continue` or `remove`.
  - Models with residual channels predict from cycle 6 onward (§1.3 item 5), so no removal happens before cycle 6.
  - The headline runs on the pooled N-CMAPSS test units with unit-level intervals. DS02 results are shown with the §6 window-level caveat.
- **Costs (illustrative, configurable, stated as assumptions in README):** planned removal C_p = 1.0 unit; unscheduled in-service failure C_f = 10 × C_p; wasted remaining life cost per cycle removed early = C_p / typical_life.
- **Lead time** L cycles (e.g., 5): a `remove` decision at cycle t takes effect at removal cycle r = t + L. Let t_fail be the engine's last cycle.
  - If r ≥ t_fail, the engine counts as an unscheduled failure.
  - Otherwise wasted life = t_fail − r.
- **Tuning uses out-of-fold predictions only.** Every threshold or margin is tuned on OOF dev-unit predictions (§1.3 item 6), never on in-sample predictions or on test units. The fixed interval T* is tuned on dev-unit lifetimes.
- **Policies:**
  1. Fixed interval (remove at cycle T*),
  2. Point-estimate threshold (remove when ŷ ≤ L + margin, margin tuned on OOF predictions),
  3. **Conformal lower-bound threshold** (remove when lower bound ≤ L),
  4. Oracle: decides at t_fail − 2L, so removal happens **exactly at t_fail − L**. That gives zero failures and wasted life of exactly L per engine — the floor.
- **Output:**
  - A table of cost/engine, failure rate and mean wasted cycles.
  - A cost-vs-α curve showing the safety/efficiency trade-off, with α ∈ {0.05, 0.10, 0.20}, split per the §6 rule. Headline numbers are at α = 0.10.
  - Sensitivity to the C_f/C_p ratio (5, 10, 50).
- The resume number comes from here: "policy 3 cut simulated maintenance cost by X% vs fixed interval with Y% failure rate". Only use numbers the scripts produce.

---

## 8. Explainability (Phase 4–5)

- Integrated gradients (captum) on the TCN/Transformer variant **with residual channels**. Attributions are computed and reported on the residual channels `X_s − f_u(W)`, because the raw W/X_s channels mostly encode flight condition. Attribute per channel and time step for each prediction, then aggregate per sensor per unit.
- Map sensors to components (e.g., T48/T50, P50 → turbine section; T30/Ps30 → HPC/combustor) and check against the known failure mode of the subset (DS02: HPT + LPT).
  - Hypothesis: attributions concentrate on turbine-section residuals for DS02.
  - Report what is found either way. If the hypothesis fails, investigate and say so.
- Patch Transformer: attention rollout plot as a secondary visual.
- LightGBM: SHAP summary as the classical counterpart.

---

## 9. Serving and demo (Phase 6)

- **ONNX export** of the best model (+ quantile head); verify parity with PyTorch (max abs diff < 1e-4) in a test; benchmark CPU latency.
- **FastAPI** (`/health`, `/predict` taking a window of shape [T, 18] → `{rul_median, rul_lower, rul_upper, decision, top_sensors}`); pydantic schemas; request validation (shape, NaNs, ranges); `/docs` auto-generated.
  - If the served model uses residual channels, `/predict` also takes an `engine_id`. The API then loads that engine's baseline MLP (fitted on its first 5 cycles and exported to ONNX alongside the model). Unknown engines return 404.
- **Docker** image for the API (slim Python base, onnxruntime, no torch).
- **Streamlit dashboard "Fleet Control Room"** (`app/dashboard.py`):
  - Fleet table: each test engine with current cycle, predicted RUL [interval], status chip (OK / plan / pull now).
  - Engine view: replay slider + autoplay over cycles; RUL median + interval band vs true RUL; maintenance trigger line.
  - Sensor attribution bar chart for the current window; one flight's altitude/Mach profile.
  - Policy panel: pick C_f/C_p and α, see cost table update.
  - Runs from a small precomputed demo bundle — never the raw HDF5. The bundle is committed to `app/demo_data/` if it is **≤ 10 MB**. Otherwise publish it as a Hugging Face dataset and download it at app start.
- **Deploy** the dashboard on Hugging Face Spaces (free) and put the link + a GIF at the top of the README. Phase 6 first checks the current Spaces SDK options: is Streamlit still a native SDK, or does it need the Docker SDK?

---

## 10. GitHub presentation (this is half the point)

README structure (in this order):
1. Title + one-line pitch + badges (CI, Python, license, "Open in Spaces").
2. GIF of the dashboard (≤ 8 MB, 15–20 s).
3. **"Results at a glance"** table generated by `scripts/make_tables.py` from `reports/results/*.json`:
   - C-MAPSS RMSE/score (capped truth) vs re-run baselines.
   - N-CMAPSS per-cycle RMSE.
   - Unit-level coverage and width at the 90% target.
   - Decision-layer cost savings.
4. Architecture diagram (Mermaid in README: data → windows → model → conformal → policy → API/dashboard).
5. "Why this is hard" — 4 bullets: real flight profiles, distribution shift, correlated windows break naive conformal, asymmetric risk.
6. Quickstart (`make setup data train app`) — must actually work from a fresh clone.
7. Experiments and findings, including negative results: each hypothesis stated up front, followed by what happened.
8. Honest limitations: simulated data; illustrative costs; protocol not comparable to all papers; DS02 intervals are window-calibrated and illustrative only; few test engines per N-CMAPSS subset.
9. Citations (datasets + papers) and license (MIT for code; dataset under NASA terms).

Repo hygiene: conventional commits, small PRs per phase (even solo — merge via PR so the history reads well), tagged releases `v0.1` … `v1.0`, pinned issues listing next steps, GitHub topics (`predictive-maintenance`, `remaining-useful-life`, `conformal-prediction`, `pytorch`, `nasa`, `time-series`).

---

## 11. Testing strategy

- `test_splits.py`:
  - Unit sets are disjoint across train/val/calib/test, and no window spans two units.
  - OOF: every dev unit is predicted only by a model that never trained on it.
- `test_windows.py`:
  - Window count is `floor((N−T)/s)+1` per (unit, cycle) segment, or 0 if N < T.
  - No window crosses a (unit, cycle) boundary or a cruise-run gap.
  - The last window aligns to the segment end, and the RUL label is the label at window end.
  - Per-cycle prediction = median of the cycle's window predictions.
  - Test stride equals train stride.
  - The window index slices per-unit arrays; nothing is materialized.
- `test_baseline_residuals.py`: engine u's baseline is fit only on u's cycles 1–5, and never reads `hs` or other engines.
- `test_labels.py`: piecewise cap; RUL monotonically non-increasing within unit.
- `test_features_no_leak.py`: feature matrix columns ⊆ W ∪ X_s (∪ residuals); fails if any banned column appears.
- `test_metrics.py`: RMSE, Saxena score vs hand examples (d = +10 → e^1 − 1; d = −13 → e^1 − 1).
- `test_conformal.py`:
  - On synthetic exchangeable data, coverage is within ±3% of target over 200 trials (mark slow).
  - `cqr_unit` uses exactly one cycle per calibration unit.
  - The α rule returns (0.02, 0.08) iff n_cal ≥ 60, else (0.05, 0.05).
- `test_policy.py`:
  - Oracle decides at t_fail − 2L and removes at t_fail − L, with zero failures and waste of exactly L per engine.
  - A never-remove policy fails every engine.
  - Tuning functions accept OOF predictions only.
- `test_onnx_parity.py`, `test_api.py` (FastAPI TestClient).

---

## 12. Compute plan

- **Laptop/CPU:**
  - C-MAPSS everything.
  - The **full N-CMAPSS Phase 3 grid at downsample 10**: LightGBM single-task, and {TCN, TSMixer, Transformer} × {single-task, multi-task}, each × {full_flight, cruise_only}.
  - The pooled DS01–DS07 conformal runs.
- **Free GPU (Kaggle 30 h/week or Colab):** full resolution **only for the top 2 models** from the downsample-10 grid.
  - Models are ranked by DS02 **validation** per-cycle RMSE, and the ranking is fixed before any test metric is seen.
  - `scripts/kaggle_train.sh` installs the package from GitHub, **keeps Kaggle's preinstalled CUDA torch** (never the CPU wheel there), and runs a named experiment, saving results JSON + checkpoint.
- **No materialized window caches.** Store each unit as contiguous `.npy` arrays (memory-mapped read-only) plus a small window index of (unit, cycle, start offset), cached by config hash. The Dataset slices windows on the fly. Why: at T = 256 and stride 10, materialized windows duplicate every sample ~25×, which is ~10–17 GB per config for DS02 alone.
- **Torch:** CPU-only wheels from the PyTorch CPU index, locally and in CI. In `pyproject.toml`, add a `[[tool.uv.index]]` entry `pytorch-cpu` = `https://download.pytorch.org/whl/cpu` with `explicit = true`, and set `[tool.uv.sources] torch = { index = "pytorch-cpu" }`.

---

## 13. Resume and interview kit (fill only from real results)

Resume line template:
> **AeroSentinel — Uncertainty-aware jet engine prognostics** (PyTorch, ONNX, FastAPI, Streamlit) · [GitHub] · [Live demo]
> - Built RUL models (TCN, Patch Transformer, LightGBM) on NASA N-CMAPSS real-flight turbofan data; RMSE **{x}** cycles on held-out engines, validated across unseen flight classes.
> - Added unit-aware conformalized quantile regression giving **{c}%** empirical per-cycle coverage at a 90% target on held-out engines (C-MAPSS FD002/FD004, pooled N-CMAPSS); naive window-level calibration differed by **{u}** pts.
> - Designed a cost-aware maintenance policy that reduced simulated maintenance cost by **{s}%** vs fixed-interval scheduling; served via ONNX (p95 **{ms}** ms CPU) behind FastAPI with a live dashboard.

Interview questions to be ready for: why unit-level splits; why cap RUL; why asymmetric score; what exchangeability means and why windows violate it; why one random cycle per calibration unit, and why DS02 alone can't support unit-level conformal; why conformal over MC dropout; how costs change the optimal α; what changes with real (non-simulated) data — sensor drift, missing data, censored run-to-failure labels.

---

## 14. Out of scope
Physics-based simulation of our own, real airline data, federated learning, LLM features, React/JS frontends. Don't add them.
