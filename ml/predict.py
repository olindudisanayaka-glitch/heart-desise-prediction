"""
================================================================================
SIMPLE DEPLOYMENT SCRIPT - Heart Disease Prediction
================================================================================
Loads the best tuned pipeline (saved by heart_disease_prediction.py) and uses
it to predict heart disease risk for a NEW patient.

Because the saved object is the FULL pipeline (preprocessing + feature
selection + tuned classifier), you only need to provide the raw clinical
values below - no manual scaling/encoding required.

Usage:
    python predict.py            -> runs on the example patient below
    python predict.py --interactive   -> asks you to type in the 13 values
================================================================================
"""

import sys
import joblib
import pandas as pd

MODEL_PATH = "outputs/best_model.pkl"
NAME_PATH = "outputs/best_model_name.pkl"

FEATURE_ORDER = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca",   # numeric
    "sex", "fbs", "exang",                                   # binary
    "cp", "restecg", "slope", "thal",                        # nominal
]

FEATURE_HELP = {
    "age": "Age in years",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dl)",
    "thalach": "Max heart rate achieved",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "ca": "Number of major vessels coloured by fluoroscopy (0-4)",
    "sex": "Sex (1 = male, 0 = female)",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "cp": "Chest pain type (0-3)",
    "restecg": "Resting ECG result (0-2)",
    "slope": "Slope of peak exercise ST segment (0-2)",
    "thal": "Thalassemia result (0-3)",
}

# Example patient (used when run without --interactive)
EXAMPLE_PATIENT = {
    "age": 58, "trestbps": 140, "chol": 289, "thalach": 145, "oldpeak": 1.6, "ca": 0,
    "sex": 1, "fbs": 0, "exang": 1,
    "cp": 2, "restecg": 1, "slope": 1, "thal": 2,
}


def load_pipeline():
    try:
        model = joblib.load(MODEL_PATH)
        try:
            best_name = joblib.load(NAME_PATH)
        except FileNotFoundError:
            best_name = "Best Model"
        return model, best_name
    except FileNotFoundError:
        raise SystemExit(
            "Could not find a saved model. Run 'python heart_disease_prediction.py' "
            "first to train, tune and save the best model."
        )


def get_patient_interactively() -> dict:
    print("Enter the patient's clinical values:")
    patient = {}
    for feat in FEATURE_ORDER:
        while True:
            raw = input(f"  {feat} ({FEATURE_HELP[feat]}): ").strip()
            try:
                patient[feat] = float(raw)
                break
            except ValueError:
                print("  Please enter a numeric value.")
    return patient


def predict(patient: dict, model, best_name: str) -> None:
    X_new = pd.DataFrame([patient], columns=FEATURE_ORDER)
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0, 1]

    print(f"\nModel used: {best_name}")
    print("Patient input:", patient)
    if pred == 1:
        print(f"\nPrediction: HEART DISEASE LIKELY (probability = {proba:.2%})")
    else:
        print(f"\nPrediction: Heart disease NOT likely (probability = {proba:.2%})")
    print("\nNote: this is a coursework demo model, NOT a medical diagnostic tool.")


def main():
    model, best_name = load_pipeline()
    if "--interactive" in sys.argv:
        patient = get_patient_interactively()
    else:
        print("Running on example patient (use --interactive to enter your own values).")
        patient = EXAMPLE_PATIENT
    predict(patient, model, best_name)


if __name__ == "__main__":
    main()
