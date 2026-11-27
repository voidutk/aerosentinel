# Research notes — sources behind the spec

Verified on 27 Nov 2026. Claude Code: use these links instead of guessing URLs.

## Why not the asteroid classifier
NASA's "potentially hazardous asteroid" flag is a fixed rule, not a hidden pattern: an asteroid is a PHA if its Earth Minimum Orbit Intersection Distance (MOID) ≤ 0.05 au **and** its absolute magnitude H ≤ 22.0 (≈140 m). NeoWs returns size estimates derived from H plus orbital data, so a classifier just re-learns two thresholds. Any interviewer who knows this will ask "what did the model learn that the definition didn't already say?" — with no good answer.
- https://cneos.jpl.nasa.gov/about/neo_groups.html
- https://cneos.jpl.nasa.gov/neo/groups.html

## Datasets
- NASA PCoE Data Repository (both datasets, citations): https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/
  - C-MAPSS (item 6): https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip
  - N-CMAPSS (item 17): https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip
  - C-MAPSS mirror: https://data.nasa.gov/docs/legacy/CMAPSSData.zip (catalog: https://catalog.data.gov/dataset/cmapss-jet-engine-simulated-data)
- N-CMAPSS data descriptor (open access): Arias Chao, Kulkarni, Goebel, Fink. *Data* 2021, 6(1), 5. https://www.mdpi.com/2306-5729/6/1/5
- N-CMAPSS preprint with DS02 unit IDs and variable tables: https://ntrs.nasa.gov/api/citations/20205001125/downloads/Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf
- PHM Society 2021 Data Challenge description (scoring = 0.5·RMSE + 0.5·NASA score, flight classes, variable list): https://data.phmsociety.org/wp-content/uploads/sites/9/2021/08/2021_Data_Challenge.pdf

## Benchmarks and protocol
- CruiseBench (arXiv 2607.19380, July 2026): cruise-stage N-CMAPSS benchmark; shows protocol choices (subsets, downsampling ×1–×1000, windows, caps, features) vary so much that N-CMAPSS results aren't one leaderboard. Fixed protocol: W + X_s inputs, T=256 native windows, stride 10, dataset-wise caps. Baselines: TSMixer avg RMSE 3.46 ± 1.71, TCN 3.89 ± 1.81 (η=5%). Code: https://github.com/NostalgiaJohn/CruiseBench — paper: https://arxiv.org/pdf/2607.19380
- PHM21 challenge solutions: winner used a deep CNN handling variable-length input; 3rd place a stacked CNN (arXiv 2111.12689); 2nd place DeVol et al. with inception-style CNN (PHM Society papers).
- C-MAPSS comparison tables (ballpark only): arXiv 2511.19124 Table I; MDPI Sensors 25(2):432 (2025) reporting 11.02 / 13.25 / 11.06 / 18.26 RMSE on FD001–4; arXiv 2104.05049 Table 4 (DA-CNN 11.78 / 16.95 / 11.56 / 18.23).
- Labeling: piecewise-linear RUL capped at 125 for C-MAPSS (Heimes 2008; Li et al. 2018).
- NASA scoring function origin: Saxena, Goebel, Simon, Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08.

## Uncertainty
- Javanmardi & Hüllermeier, "Conformal Prediction Intervals for Remaining Useful Lifetime Estimation", IJPHM 14(2), 2023 — https://arxiv.org/abs/2212.14612. Notes: CP wraps any point predictor; finite-sample coverage under exchangeability; open problem for the non-exchangeable case (our unit-aware calibration addresses this practically).
- "Remaining Useful Life Estimation for Aircraft Engines with Risk-Aware Prediction Intervals via Conformalized Quantile Regression", IJPHM, published 27 Apr 2026 — asymmetric miscoverage between lower and upper bounds. https://www.papers.phmsociety.org/index.php/ijphm/article/view/4724
- Romano, Patterson, Candès (2019), Conformalized Quantile Regression.

## Domain shift / related ideas worth citing
- Domain adaptation across N-CMAPSS flight classes on DS03 (arXiv 2302.01704).
- Health-index estimation on N-CMAPSS DS03 using ground-truth HI for evaluation only (arXiv 2405.04990).
- Unified fault detection + isolation + RUL on N-CMAPSS (IJPHM, "Flux" paper).

## Claude Code workflow (official docs)
- Best practices (verification checks, plan mode, CLAUDE.md, subagents, /clear): https://code.claude.com/docs/en/best-practices
- Memory / CLAUDE.md: https://code.claude.com/docs/en/memory
- Hooks guide (PreToolUse/PostToolUse/Stop, exit code 2 blocks, stop_hook_active): https://code.claude.com/docs/en/hooks-guide
- Subagents: https://code.claude.com/docs/en/sub-agents
- Skills: https://code.claude.com/docs/en/skills
