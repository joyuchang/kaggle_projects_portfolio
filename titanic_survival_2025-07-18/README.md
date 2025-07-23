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
│   ├── titanic_model_v2_2025-07-22_WIP.ipynb  # Second version with feature engineering and multiple model testing
│   ├── eda_titanic_v1_based_on_raw_data_2025-07-21.ipynb  # EDA notebook based on raw data
│   └── eda_titanic_v2_feature_analysis_2025-07-22.ipynb   # EDA notebook for feature analysis
├── output/
│   └── titanic_submission_v1_2025-07-21.csv   # Submission file
└── README.md                                  # This file

```

## Progress Overview - *first attempt - July 2025*
> Based on Kaggle's official Titanic tutorial, then extended with my own EDA and model refinement.
- Baseline model (`titanic_model_v1.ipynb`) built using **TensorFlow Decision Forests (TFDF)**  
- Followed Kaggle's official tutorial as initial guidance  
- Implemented ensemble strategy (100 trees, varied seeds)  
- Generated and submitted first prediction file: `submission.csv`  
- Kaggle Public Score: **0.80622**  
- Completed **EDA** on original dataset
- Created `titanic_model_v1_clean.ipynb` for Kaggle submission (lightweight, no output)

## Key EDA insights  
- **Sex** and **Pclass** are highly correlated with survival  
    → Females and 1st-class passengers had higher survival rates  
- **Fare** showed a moderate positive correlation  
    → Higher fare, slightly higher survival rate  
- **Age** showed weak correlation  
    → Younger passengers did not significantly influence survival  
- Visualization: count plots, histograms, heatmaps, and bar plots  

## Next steps
- Add **feature engineering** (e.g., `Title`, `FamilySize`, `IsAlone`)  
- Try alternative models (XGBoost, RandomForest)  
- Apply validation techniques to avoid overfitting  
- Optimize model and improve Kaggle score  

## Notes
This project is based on the [Kaggle Titanic competition].  
The first attempt (`v1`) was built following the official tutorial.  
I will continue interating and deepening my understanding of ML through this case.  