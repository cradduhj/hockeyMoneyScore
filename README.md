# hockeyMoneyScore

**Do NHL players' on-ice performance metrics actually predict what they get paid?**

This project builds composite performance scores — a per-game **local MoneyScore** and a
per-season **global MoneyScore** — from NHL performance data, then tests how well those
scores (and the underlying stats) explain a player's cap hit as a percentage of the league
salary cap. An Elo rating system is layered on top of local MoneyScore to produce head-to-head
player rankings within a season.

SLM 418 (Miami University) final project by Harrison Cradduck and Will Paz, presented at the
Joint Mathematics Meetings (JMM) 2026.

## Approach

1. **Data collection** (`Web Scraping/`) — player performance stats pulled from the NHL API and
   player financial/cap-hit data scraped separately, both at the game and season level.
2. **Player data merge** (`Player Data/`) — R (`tidyverse`, `fuzzyjoin`) notebooks that clean,
   normalize, and merge the performance and financial datasets by season/player, handling name
   mismatches between sources.
3. **Feature selection & MoneyScore construction** (`Feature Selection/`) — for both the
   game-level and season-level datasets, five feature-selection methods (Extra Trees, XGBoost,
   Lasso, Recursive Feature Elimination, Ridge) each vote on which stats matter; features with
   ≥3 votes are kept, scaled to [0, 1], and combined into a single weighted score using
   XGBoost-derived feature importances as weights — `localMoneyScore` per game, `globalMoneyScore`
   per season.
4. **Elo ratings** (`Elo/`) — treats every pair of players in a game as a head-to-head matchup
   decided by `localMoneyScore`, and runs a standard Elo update (tunable via `eloTuning.py`,
   which grid-searches the K-factor and score-margin multiplier for prediction accuracy) to
   produce a running player rating per season.
5. **Final predictive model** (`Predictions/moneyScoreFinalModel.py`) — an XGBoost regressor
   (tuned via grid search + 5-fold CV) predicting a player's cap-hit percentage from the
   selected performance features, evaluated with RMSE/MAE/R² and inspected via feature
   importance plots.

## Repo structure

```
Web Scraping/       # Raw API/financial data pulls
Player Data/         # R: merges performance + financial data by player/season
Feature Selection/    # Python: voting-based feature selection -> local/global MoneyScore
Elo/                 # Python: Elo ratings built on localMoneyScore
Predictions/          # Python: final XGBoost cap-hit model + feature importance
Data/                 # Intermediate/processed data (zipped)
```

## Running this

Python side (`Feature Selection/`, `Elo/`, `Predictions/`) needs:
```
pandas numpy scikit-learn xgboost matplotlib seaborn
```

R side (`Player Data/`) needs:
```r
install.packages(c("tidyverse", "fuzzyjoin"))
```

The `Player Data/` notebooks read from a data directory controlled by the
`HOCKEY_MONEYSCORE_DATA_DIR` environment variable (containing `API Data/`, `Finance Data/`, and
`Season Data/` subfolders matching the raw exports) — set that instead of editing the notebooks.
Pipeline order: `Player Data/` (merge raw data) →
`Feature Selection/` (build MoneyScore) → `Elo/` (rate players) and
`Predictions/moneyScoreFinalModel.py` (fit the cap-hit model) can run independently once
MoneyScore is built.

## Results

Feature importance plots from the voting-based selection are in `Feature Selection/`
(`globalMoneyScoreFeatureImportance.png`, `localMoneyScoreFeatureImportance.png`); the final
model's evaluation metrics (RMSE/MAE/R²) and actual-vs-predicted plot are produced by
`Predictions/moneyScoreFinalModel.py`.
