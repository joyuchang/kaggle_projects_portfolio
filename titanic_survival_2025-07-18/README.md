# Titanic - Machine Learning from Disaster
This is my **first Kaggle project**, exploring survival prediction on the Titanic using machine learning.
The project follows a step-by-step structure including data exploration, baseline modeling, and furture improvements.

## Project Structure
```
/titanic_survival_2025-07-18/
├── data/                                      # Raw competition data (ignored in .git)
├── notebooks/
│   ├── titanic_model_v1_2025-07-21.ipynb      # First version model using TFDF + ensemble
│   ├── titanic_model_v1_clean.ipynb           # Cleaned version for Kaggle submission
│   ├── titanic_model_v2_2025-07-24.ipynb      # Second version with feature engineering and multiple model testing
│   ├── titanic_model_v3_2025-07-31.ipynb      # Third version: added Age + XGBoost pipeline
│   ├── titanic_model_v3_xgb_tuning_2025-07-31.ipynb  # V3 tuning-only notebook
│   ├── eda_titanic_v1_based_on_raw_data_2025-07-21.ipynb  # EDA notebook based on raw data
│   └── eda_titanic_v2_feature_analysis_2025-07-22.ipynb   # EDA notebook for feature analysis
├── output/
│   ├── titanic_submission_v1_2025-07-21.csv           # 1st submission file
│   ├── titanic_submission_v2_xgboost_2025-07-24.csv   # 2nd submission file
│   ├── titanic_submission_v2_ensemble_2025-07-24.csv  # 3rd submission file
│   ├── titanic_submission_v3-a_xgb_tuning_2025-07-31.csv         # 4th submission file
│   ├── titanic_submission_v3-b_voting_ensemble_2025-07-31.csv    # 5th submission file
│   └── titanic_submission_v3-c_stacking_ensemble_2025-07-31.csv  # 6th submission file
└── README.md                                  # This file
```

## Progress Overview

### Model V1 (2025-07-21)
> Based on Kaggle's official Titanic tutorial, then extended with my own EDA and model refinement.
- Baseline model (`titanic_model_v1.ipynb`) built using **TensorFlow Decision Forests (TFDF)**  
- Followed Kaggle's official tutorial as initial guidance  
- Implemented ensemble strategy (100 trees, varied seeds)  
- Generated and submitted first prediction file: `submission.csv` (which is `titanic_submission_v1_2025-07-01.csv`)  
- **Public Kaggle Score: 0.80622**  
- Best score so far, but logic interpretability is limited.

### Model V2 (2025-07-24)
> Refined model with full EDA + feature engineering + model comparison.
- Performed detailed EDA (v1: raw features / v2: engineered features)
- Added features: `Title_Grouped`, `FamilySize`, `IsAlone`, `FarePerPerson`, `FamilyGroup`
- Tested 4 models: Logistic Regression, Decision Tree, Random Forest, XGBoost
- Selected XGBoost as final model
- **Kaggle Scores:**  
  - `titanic_submission_v2_xgboost_2025-07-24.csv`: **0.74401**
  - `titanic_submission_v2_ensemble_2025-07-24.csv`: **0.74641**
- As a begineer in machine learning, I actively used ChatGPT and some Perplexity for support during Model V2 development, especially for structuring feature analysis, comparing models. Through this interacitve process, I was able to learn a great deal and deepen my understanding of apllied ML workflows.

### Model V3 (2025-07-31)
> Improved model with `Age` (groupwise imputation), consistent pipeline, and tuned XGBoost via GridSearchCV.
- Added new feature: `Age` **(Imputed by `Title_Grouped` median)**
- Continued use of engineered features: `Title_Grouped`, `FamilySize`, `IsAlone`, `FarePerPerson`
- Built full pipeline including scaling + one-hot encoding
- Used **GridSearchCV** with 9 combinations of XGBoost parameters
- Trained and submitted two models:
  - `titanic_model_v3_2025-07-31.ipynb` → submission file V3-a
  - `titanic_model_v3_voting_2025-07-31.ipynb` → submission file V3-b
  - `titanic_model_v3_stacking_2025-07-31.ipynb` → submission file V3-c
- Kaggle Score:
  - `titanic_submission_v3-a_xgb_tuning_2025-07-31.csv`: 0.75358
  - `titanic_submission_v3-b_voting_ensemble_2025-07-31.csv`: 0.77751
  - `titanic_submission_v3-b_stacking_ensemble_2025-07-31.csv`: 0.77272


### Submission Comparison Table
| Version | File Name | Features Used | Model | Tuning | CV Accuracy | Kaggle Public Score | Notes |
| :- | :- | :- | :- | :- | :- | :- | :- |
| **V1** | `titanic_submission_v1_2025-07-21.csv` | Raw: `Sex`, `Pclass`, `Fare` | TFDF (ensemble) | No | N/A | **0.80622** | Best so far, simple ensemble strategy |
| **V2-a** | `titanic_submission_v2_xgboost_2025-07-24.csv` | Engineered: `Title_Grouped`, `IsAlone`, `FamilySize`, `FarePerPerson`, `FamilyGroup` | XGBoost | No | 0.8249 | 0.74401 | Strong CV, overfit suspected |
| **V2-b** | `titanic_submission_v2_ensemble_2025-07-24.csv` | Same as V2-a | XGB + others (soft-vote) | No | \~0.826 | 0.74641 | Slight improvement from voting |
| **V3-a** | `titanic_submission_v3-a_xgb_tuned_2025-07-31.csv` | V2 features + `Age` (imputed) | XGBoost | GridSearchCV | 0.9293 | 0.75358 | Main baseline for V3 |
| **V3-b** | `titanic_submission_v3-b_voting_ensemble_2025-07-31.csv` | V3 features | VotingClassifier | No | 0.8316 | 0.77751 | Combine multiple algorithms |
| **V3-c** | `titanic_submission_v3-c_stacking_ensemble_2025-07-31.csv` | V3 features | StackingClassifier | No / Light | 0.8271 | 0.77272 | Deep ensemble, meta-level fusion |


## Key EDA insights  
- `Sex`, `Pclass`, `Title_Grouped` were **strong predictors**
- Engineered features like `IsAlone` and `FarePerPerson` added **moderate value**
- `Deck` and `Embarked` features were **excluded** due to redundancy or noise
- SHAP & XGBoost importance confirmed the dominance of `Title`, `Fare`, `Pclass`

## Learning Highlights
- Built custom pipeline for **feature engineering** and **model evaluation**
- Learned how to apply:
  - One-hot encoding
  - Cross-validation (CV)
  - Model comparison
  - SHAP interpretability
  - Submission generation & testing
- Understood the **gap between CV score and Kaggle leaderborad**
  - CV Accuracy (XGBoost): **0.8249**
  - Kaggle Public Score: **0.74401**

## Next steps (Model V3 Plan)
- Add `Age` (with imputation) into the model
- Tune XGBoost with `GridSearchCV` or `RandomizedSearchCV`
- Try **stacking** and advanced ensembling
- Submit more versions to monitor generalization

## Notes
This project is based on the [Kaggle Titanic competition].  
I aim to use this project as a foundation for future ML/DS learning and portfolio building.
2025-07-31