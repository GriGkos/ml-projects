# Elo Merchant Category Recommendation

End-to-end regression project for the Kaggle competition [Elo Merchant Category Recommendation](https://www.kaggle.com/competitions/elo-merchant-category-recommendation).

## Goal

Predict the `target` loyalty score from card attributes and two transaction tables. The competition metric is RMSE, so lower is better.

## Validation and result

Validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` by the indicator `target < -30`: each validation fold contains 441-442 extreme objects. All model selection and ensemble weights are based on out-of-fold predictions. The final LightGBM configuration is additionally checked on an independent split with `random_state=2026`.

| Experiment | OOF RMSE |
| --- | ---: |
| DummyRegressor | 3.85050 |
| Ridge on raw card features | 3.84454 |
| LightGBM with Optuna tuning | 3.65100 |
| CatBoost with native categorical features | 3.65271 |
| XGBoost, three-seed bagging | 3.65008 |
| XGBoost with cross-fitted target encodings | 3.65171 |
| Weighted OOF blend | 3.64705 |
| **Stacked blend with cross-fitted extreme-regime adjustment** | **3.64665** |

The final blend uses LightGBM, CatBoost, a three-seed XGBoost bag, and XGBoost trained on the expanded feature set. The target-encoding component is retained as a documented negative experiment: in the current OOF optimization its weight is zero. A regularized second-level Ridge model uses cross-fitted OOF predictions. The extreme `target < -30` regime is handled by a separate classifier; its threshold, replacement value and continuous correction are selected inside the other folds.

The uploaded `submission_final.csv` received private RMSE **3.61471** and public RMSE **3.70359**. In the downloaded private leaderboard this corresponds to approximately rank 787 of 4,111 (top 19.1%). The top-10% private threshold is 3.61369, so the remaining gap is 0.00102.

## Main findings

- The target has an unusual spike near `-33.22`: 2,207 train objects (1.09%) belong to this extreme regime.
- Transaction aggregates are more useful than the three raw card features: compact LightGBM ablation improves from `3.84196` to `3.65499` RMSE.
- More features did not improve the compact LightGBM ablation (`v1 3.65499`, `v2 3.65506`), but expanded XGBoost receives a non-zero OOF-selected blend weight.
- Models have highly correlated residuals, so the ensemble improvement is modest and is reported conservatively through OOF RMSE.

## Rejected hypotheses

- The expanded v2 feature set does not improve the compact LightGBM ablation by itself (`3.65506` versus `3.65499`), so its use is justified only by the blend's OOF weights.
- Cross-fitted target encoding does not improve the matched XGBoost model and receives an OOF weight of zero.
- Outlier correction reduces RMSE on `target < -30`, but slightly worsens it on the remaining cards; the final correction is deliberately conservative and selected inside folds.

## References and authorship

- [Competition overview](https://www.kaggle.com/competitions/elo-merchant-category-recommendation) and the [public code collection](https://www.kaggle.com/competitions/elo-merchant-category-recommendation/code) were used to understand the data format and common validation pitfalls.
- The implementation, feature aggregation, cross-fitted target encoding, validation, blending and report are written in this repository. Public solutions were not copied.

## Project layout

```text
03_kaggle_feature_ensemble/
  notebooks/solution.ipynb
  submissions/submission_final.csv
  reports/kaggle_submission.png  # Kaggle submission evidence
```

The repository intentionally contains no raw data, caches, alternate submissions, or helper scripts: `solution.ipynb` is self-contained and includes the feature engineering and training code.

## Reproduction

1. Download the competition data and place all CSV files in a local `data/` directory next to the notebook. The directory is ignored by Git.
2. Install dependencies:

```powershell
cd projects/03_kaggle_feature_ensemble
pip install -r requirements.txt
```

3. Open and run `notebooks/solution.ipynb` from top to bottom. The complete feature code, model fitting, Optuna search, OOF blending and submission creation are inside the notebook.
4. The first feature build and random-card EDA sample read the large transaction files; expect roughly 45-60 minutes on this computer and at least 8 GB RAM. Subsequent runs use local caches in `artifacts/`.
5. Upload `submissions/submission_final.csv` to Kaggle and save the resulting screenshot with username and score in `reports/`.

`reports/README.md` specifies the expected screenshot name and contents.

`data/` and `artifacts/` are intentionally excluded from Git because they contain source files and large generated outputs.
