# Kaggle Projects Portfolio
This repository showcases my Kaggle machine learning projects with structured notebooks and clear project management.  

## Repository Structure
Each Kaggle project is stored in a seperate subdirectory, following this convention:  

```
/kaggle_projects_portfolio/
├── titanic_survival_2025-07-18/ # Titanic survival prediction project (v1~v2)
│ ├── data/ # Raw data (not included in version control)
│ ├── notebooks/ # EDA, modeling, feature engineering notebooks
│ ├── output/ # Kaggle submission files
│ └── README.md # Project-specific README
│
├── <future_project_name>/ # Placeholder for upcoming Kaggle projects
│ ├── data/
│ ├── notebooks/
│ ├── output/
│ └── README.md
│
└── README.md # Main portfolio README
```

Each project folder contains:
- `data/` → raw CSV files (ignored in `.gitignore`)  
- `notebooks/` → Jupyter Notebooks for EDA and modeling  
- `output/` → submission `.csv` files
- `README.md` → project summary and documentation


## projects **[WIP]**  
| Projects | Description | Start Date | Best Kaggle Score |
|---|---|---|---|
|[Titanic Survival](titanic_survival_2025-07-18/README.md)|Predicting survival on the Titanic using EDA + TFDF + feature engineering|2025-07-18|0.80622|
> The Titanic project has multiple model versions submitted:
> - V1: TFDF ensemble — 0.80622  
> - V2: XGBoost — 0.74401  
> - V2: Ensemble (XGBoost + Logistic) — 0.74641  

## Stack Used
- **Language:** Python  
- **Core libraries:** Pandas, NumPy, Seaborn, Matplotlib  
- **Modeling:**
  - TensorFlow Decision Forests (TFDF)
  - Scikit-learn (Logistic Regression, Tree, Random Forest)
  - XGBoost, LightGBM
  - SHAP (model explainability)
- **Tools:** Jupyter Notebook  

## Structure
Each project contains:
1. **EDA (Exploratory Data Analysis)**  
2. **Feature Engineering**  
3. **Model Building & Evaluation**  
4. **SHAP Interpretation**  
5. **Model Ensemble or Optimization**  
6. **Submission to Kaggle**  
7. **Version Tracking with File Naming**

## Notes
- All projects are version-controlled with explicit file naming (e.g., `model_v2_2025-07-24.ipynb`)
- Models are evaluated using both cross-validation and Kaggle public scores
- Some projects are built with the assistance of AI (e.g., ChatGPT), especially for structuring code. All logicand modeling decisions are reviewed and implemented by myself.

Thanks for visiting.  
I'm currently working on **Model V3** for Titanic, and new datasets will be added soon.
2025-07-24