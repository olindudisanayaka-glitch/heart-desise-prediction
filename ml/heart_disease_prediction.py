"""
================================================================================
HEART DISEASE PREDICTION - COMPLETE MACHINE LEARNING PIPELINE
================================================================================
Module      : Machine Learning (HNDSE)
Problem type: Binary Classification (Heart Disease: Present = 1, Absent = 0)

This script implements the FULL pipeline required by the coursework:
  1. Problem Definition
  2. Data Collection
  3. Data Exploration & Preprocessing (EDA, missing values, encoding, scaling)
  4. Feature Engineering (feature selection)
  5. Model Training (Logistic Regression, KNN, SVM, Random Forest)
  6. Hyperparameter Tuning (GridSearchCV)
  7. Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix)
  8. Model Comparison (best model selection + justification)
  9. Simple Deployment (best pipeline saved with joblib -> used by predict.py)
 10. Interpretation & Insights (feature importance)

Run:
    python heart_disease_prediction.py

Outputs (saved into ./outputs/):
    - eda_*.png                 (exploratory plots)
    - confusion_matrix_*.png    (per model)
    - roc_curves.png
    - model_comparison.png
    - feature_importance.png
    - model_comparison_table.csv
    - best_model.pkl            (deployable pipeline used by predict.py)
================================================================================
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless / script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
RANDOM_STATE = 42

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

LOCAL_PATH = "heart.csv"
DATA_URL = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"


# ==============================================================================
# 1. PROBLEM DEFINITION
# ==============================================================================
# Problem  : Predict whether a patient has heart disease based on clinical
#            measurements (age, blood pressure, cholesterol, ECG results, etc.)
# Type     : Supervised Binary Classification (target: 0 = no disease, 1 = disease)
# Why it matters: Cardiovascular disease is a leading cause of death worldwide.
#            An early, low-cost screening model built from routine clinical
#            measurements can help prioritise patients for further diagnostic
#            tests (e.g. angiography) and support clinicians' decision-making.
# ==============================================================================


# ==============================================================================
# 2. DATA COLLECTION
# ==============================================================================
def load_data() -> pd.DataFrame:
    """
    Loads the UCI Cleveland Heart Disease dataset (303 patients, 13 clinical
    features + 1 binary target column 'target').

    Source : UCI Machine Learning Repository - Heart Disease Data Set
              (Janosi, Steinbrunn, Pfisterer & Detrano, 1989), redistributed as
              a clean CSV at the URL below.
    Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
              oldpeak, slope, ca, thal
    Target  : target (0 = no heart disease, 1 = heart disease present)
    """
    if os.path.exists(LOCAL_PATH):
        print(f"[Data] Loading dataset from local file: {LOCAL_PATH}")
        df = pd.read_csv(LOCAL_PATH)
    else:
        try:
            print(f"[Data] Downloading dataset from: {DATA_URL}")
            df = pd.read_csv(DATA_URL)
            df.to_csv(LOCAL_PATH, index=False)
        except Exception as e:
            raise SystemExit(
                f"Could not download the dataset ({e}).\n"
                f"Please download 'heart.csv' manually (UCI Heart Disease "
                f"dataset, 303 rows / 14 columns) and place it in this folder."
            )
    print(f"[Data] Shape: {df.shape}")
    return df


NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
NOMINAL_FEATURES = ["cp", "restecg", "slope", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + BINARY_FEATURES + NOMINAL_FEATURES
TARGET = "target"


# ==============================================================================
# 3. DATA EXPLORATION (EDA)
# ==============================================================================
def explore_data(df: pd.DataFrame) -> None:
    print("\n[EDA] First rows:\n", df.head())
    print("\n[EDA] Info:")
    df.info()
    print("\n[EDA] Missing values per column:\n", df.isnull().sum())
    print("\n[EDA] Class balance (target):\n", df[TARGET].value_counts())
    print("\n[EDA] Summary statistics:\n", df.describe())

    # Target class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x=TARGET, data=df, palette="Set2")
    plt.title("Class Balance: Heart Disease (1) vs No Disease (0)")
    plt.xlabel("Target")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_class_balance.png", dpi=150)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_correlation_heatmap.png", dpi=150)
    plt.close()

    # Distributions of key numeric features
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, col in zip(axes.flatten(), NUMERIC_FEATURES):
        sns.histplot(data=df, x=col, hue=TARGET, kde=True, ax=ax, palette="Set2")
        ax.set_title(col)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/eda_numeric_distributions.png", dpi=150)
    plt.close()

    print(f"\n[EDA] Plots saved to '{OUT_DIR}/'.")


# ==============================================================================
# 3b. PREPROCESSING (handled inside an sklearn Pipeline to avoid data leakage)
# ==============================================================================
def build_preprocessor() -> ColumnTransformer:
    """
    Justification:
      - Numeric features: median imputation (robust to outliers, e.g. cholesterol
        spikes) + StandardScaler (required by distance-based models like KNN/SVM
        and improves convergence for Logistic Regression).
      - Binary features (already 0/1): most-frequent imputation only; no scaling
        needed since they are already on a comparable 0-1 scale.
      - Nominal categorical features (cp, restecg, slope, thal): most-frequent
        imputation + One-Hot Encoding, since these are unordered categories and
        encoding them as raw integers would falsely imply an order/magnitude.
    """
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    binary_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])
    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("bin", binary_pipe, BINARY_FEATURES),
        ("nom", nominal_pipe, NOMINAL_FEATURES),
    ])
    return preprocessor


# ==============================================================================
# 4. FEATURE ENGINEERING - Feature Selection
# ==============================================================================
# Technique: SelectKBest with the ANOVA F-test (f_classif).
# Why     : After one-hot encoding we have ~19 columns. Not all are equally
#           predictive (e.g. fasting blood sugar 'fbs' is known to be weak).
#           SelectKBest ranks features by their statistical relationship with
#           the target and keeps only the top-K, which reduces noise/overfitting,
#           speeds up training, and improves interpretability for clinicians.
# ==============================================================================
N_SELECTED_FEATURES = 10


def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    """Recovers readable feature names after the ColumnTransformer step."""
    return preprocessor.get_feature_names_out()


def make_full_pipeline(classifier) -> Pipeline:
    """Combines preprocessing + feature selection + classifier into ONE pipeline.
    This single object is what gets cross-validated, tuned, evaluated AND
    deployed (saved with joblib) -- so the exact same preprocessing is applied
    automatically to new patient data at prediction time."""
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("select", SelectKBest(score_func=f_classif, k=N_SELECTED_FEATURES)),
        ("clf", classifier),
    ])


# ==============================================================================
# 5 & 6. MODEL TRAINING + HYPERPARAMETER TUNING (GridSearchCV)
# ==============================================================================
# 4 algorithms used (coursework requires min 3 / max 4, covering all 3 types):
#   - Logistic Regression  -> basic / linear model
#   - K-Nearest Neighbors  -> distance-based model
#   - Support Vector Machine (SVC) -> distance-based model
#   - Random Forest        -> ensemble model
# ==============================================================================
MODEL_GRID = {
    "Logistic Regression": {
        "pipeline": make_full_pipeline(LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
    },
    "KNN": {
        "pipeline": make_full_pipeline(KNeighborsClassifier()),
        "params": {
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__p": [1, 2],
        },
    },
    "SVM": {
        "pipeline": make_full_pipeline(SVC(probability=True, random_state=RANDOM_STATE)),
        "params": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__kernel": ["rbf", "linear"],
            "clf__gamma": ["scale", "auto"],
        },
    },
    "Random Forest": {
        "pipeline": make_full_pipeline(RandomForestClassifier(random_state=RANDOM_STATE)),
        "params": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5],
        },
    },
}


def tune_models(X_train, y_train):
    """Runs GridSearchCV (5-fold stratified CV, scored on ROC-AUC) for every
    model and returns the best fitted pipeline for each."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_estimators, tuning_summary = {}, []

    for name, cfg in MODEL_GRID.items():
        print(f"\n[Tuning] {name} ...")
        grid = GridSearchCV(
            cfg["pipeline"], cfg["params"], cv=cv,
            scoring="roc_auc", n_jobs=-1, refit=True
        )
        grid.fit(X_train, y_train)
        best_estimators[name] = grid.best_estimator_
        tuning_summary.append({
            "Model": name,
            "Best Params": grid.best_params_,
            "Best CV ROC-AUC": round(grid.best_score_, 4),
        })
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best CV ROC-AUC: {grid.best_score_:.4f}")

    pd.DataFrame(tuning_summary).to_csv(f"{OUT_DIR}/tuning_summary.csv", index=False)
    return best_estimators


# ==============================================================================
# 7. MODEL EVALUATION
# ==============================================================================
def evaluate_model(name, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1-score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Disease", "Disease"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/confusion_matrix_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()

    print(f"\n[Evaluation] {name}")
    for k, v in metrics.items():
        if k != "Model":
            print(f"  {k}: {v:.4f}")

    return metrics, y_proba


def plot_roc_curves(results_proba: dict, y_test) -> None:
    plt.figure(figsize=(7, 6))
    for name, y_proba in results_proba.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/roc_curves.png", dpi=150)
    plt.close()


# ==============================================================================
# 8. MODEL COMPARISON
# ==============================================================================
def compare_models(results: list) -> pd.DataFrame:
    df_results = pd.DataFrame(results).sort_values("F1-score", ascending=False).reset_index(drop=True)
    df_results.to_csv(f"{OUT_DIR}/model_comparison_table.csv", index=False)
    print("\n[Comparison] Model performance summary:\n", df_results)

    plt.figure(figsize=(9, 5))
    melted = df_results.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"])
    sns.barplot(data=melted, x="Model", y="value", hue="variable")
    plt.title("Model Comparison Across Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/model_comparison.png", dpi=150)
    plt.close()

    best_row = df_results.iloc[0]
    print(
        f"\n[Comparison] Best model: {best_row['Model']} "
        f"(F1-score={best_row['F1-score']:.4f}, ROC-AUC={best_row['ROC-AUC']:.4f}).\n"
        f"Justification: it achieves the best balance between correctly detecting "
        f"true heart-disease cases (Recall) and avoiding false alarms (Precision), "
        f"summarised by the F1-score, while also ranking patients well by risk (ROC-AUC)."
    )
    return df_results


# ==============================================================================
# 10. INTERPRETATION & INSIGHTS - Feature Importance
# ==============================================================================
def plot_feature_importance(best_name, best_model) -> None:
    preprocessor = best_model.named_steps["preprocess"]
    selector = best_model.named_steps["select"]
    clf = best_model.named_steps["clf"]

    all_names = get_feature_names(preprocessor)
    selected_names = all_names[selector.get_support()]

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        title = f"Feature Importance - {best_name}"
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])
        title = f"Feature Importance (|coefficient|) - {best_name}"
    else:
        print(f"[Interpretation] {best_name} has no direct feature-importance attribute "
              f"(e.g. SVM/KNN) - skipping importance plot.")
        return

    imp_df = pd.DataFrame({"Feature": selected_names, "Importance": importances})
    imp_df = imp_df.sort_values("Importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=imp_df, x="Importance", y="Feature", palette="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=150)
    plt.close()

    print(f"\n[Interpretation] Top features driving '{best_name}' predictions:\n", imp_df.head(10))


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================
def main():
    # 2. Data Collection
    df = load_data()

    # 3. EDA
    explore_data(df)

    # Train/test split (stratified to preserve class balance) - BEFORE any
    # fitting of scalers/encoders/selectors, to prevent data leakage.
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n[Split] Train: {X_train.shape}, Test: {X_test.shape}")

    # 5 & 6. Train + tune all models (preprocessing + feature selection happen
    # inside each pipeline, refit safely within every CV fold)
    best_estimators = tune_models(X_train, y_train)

    # 7. Evaluate every tuned model on the held-out test set
    results, results_proba = [], {}
    for name, model in best_estimators.items():
        metrics, y_proba = evaluate_model(name, model, X_test, y_test)
        results.append(metrics)
        results_proba[name] = y_proba

    plot_roc_curves(results_proba, y_test)

    # 8. Compare models & select the best one
    comparison_df = compare_models(results)
    best_name = comparison_df.iloc[0]["Model"]
    best_model = best_estimators[best_name]

    # 10. Interpretation
    plot_feature_importance(best_name, best_model)

    # 9. Simple Deployment - persist the WHOLE pipeline (preprocessing +
    # feature selection + tuned classifier) so predict.py can load it and
    # run predictions directly on raw, unprocessed patient input.
    joblib.dump(best_model, f"{OUT_DIR}/best_model.pkl")
    joblib.dump(best_name, f"{OUT_DIR}/best_model_name.pkl")
    print(f"\n[Deployment] Saved best model ('{best_name}') to '{OUT_DIR}/best_model.pkl'")
    print("Run 'python predict.py' to try a live prediction.")


if __name__ == "__main__":
    main()
