# Heart Disease Prediction — ML Coursework Project

A complete supervised machine-learning pipeline that predicts whether a
patient has heart disease using the UCI Cleveland Heart Disease dataset
(303 patients, 13 clinical features).

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. Run the pipeline (trains, tunes, evaluates, saves the best model)

```bash
python heart_disease_prediction.py
```

This will:
- Download `heart.csv` automatically (or use a local copy if present)
- Run EDA and save plots to `outputs/`
- Preprocess data inside an sklearn `Pipeline` (no data leakage)
- Select the top 10 features (`SelectKBest`, ANOVA F-test)
- Train + tune 4 models with `GridSearchCV`: Logistic Regression, KNN, SVM,
  Random Forest
- Evaluate every model (Accuracy, Precision, Recall, F1, ROC-AUC, confusion
  matrices, ROC curves)
- Compare models and pick the best one
- Plot feature importance for the winning model
- Save the winning pipeline to `outputs/best_model.pkl`

## 3. Try a prediction (deployment)

```bash
python predict.py                  # uses a built-in example patient
python predict.py --interactive    # type in your own patient values
```

## 4. Mapping code to your report sections

| Report section | Where in the code |
|---|---|
| Problem Definition | Top comment block, Section 1 |
| Data Collection | `load_data()` |
| EDA & Preprocessing | `explore_data()`, `build_preprocessor()` |
| Feature Engineering | `SelectKBest` in `make_full_pipeline()` |
| Model Training | `MODEL_GRID` dictionary |
| Hyperparameter Tuning | `tune_models()` (GridSearchCV) |
| Model Evaluation | `evaluate_model()`, `plot_roc_curves()` |
| Model Comparison | `compare_models()` |
| Deployment | `predict.py` |
| Interpretation & Insights | `plot_feature_importance()` |

All generated charts/tables land in `outputs/` — paste these into your report
as figures/appendices.

## Dataset citation (Harvard format)

Janosi, A., Steinbrunn, W., Pfisterer, M. and Detrano, R. (1989) *Heart
Disease* [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C52P4X

## Important — academic integrity

This script gives you a working pipeline so you can focus on understanding
and explaining it. Your assignment requires the AI-content of the submitted
report to be 0% and each member to write their own reflection — so re-run
the code yourself, look at your own results/plots, and write the
explanations, EDA insights, and reflection sections in your own words.
