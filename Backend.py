import numpy as np
import pandas as pd
import joblib
import os

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ============================================================
# PATH
# ============================================================
DATA_PATH = r"C:\Users\DELL\Desktop\Final Data v1.xlsx"
MODEL_PATH = "slope_model.pkl"

# ============================================================
# UI → MODEL NAME MAPPING
# ============================================================
COLUMN_MAP = {
    "Groundwater": "Groundwater",   # Case renamed
    "Slope Angle (deg)": "Angle",
    "Slope Height H (m)": "H",
    "Cohesion c (kPa)": "C",
    "Friction Angle (deg)": "F. Angle",
    "Unit Weight (kN/m3)": "U.W"
}

# ============================================================
# SCALER
# ============================================================
class ManualScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# ============================================================
# TRAIN MODEL
# ============================================================
def train_and_save_model():
    print("🔵 Training model...")

    df = pd.read_excel(DATA_PATH)

    # Rename Case → Groundwater
    df = df.rename(columns={"Case": "Groundwater"})

    df['FOS'] = pd.to_numeric(df['FOS'], errors='coerce')
    df.dropna(subset=['FOS'], inplace=True)

    X = df.drop('FOS', axis=1)
    y = df['FOS']

    features = X.columns.tolist()
    print("Features used:", features)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42
    )

    scaler = ManualScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=130,
        max_depth=2,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1,
        random_state=42
    )

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    rmse_scores = np.sqrt(-cross_val_score(
        model, X_train, y_train,
        scoring='neg_mean_squared_error',
        cv=kfold
    ))

    print("CV RMSE:", rmse_scores.mean())

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nMODEL PERFORMANCE")
    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "features": features
    }, MODEL_PATH)

    print("\n✅ Model saved")

# ============================================================
# LOAD MODEL
# ============================================================
def load_model():
    if not os.path.exists(MODEL_PATH):
        train_and_save_model()

    data = joblib.load(MODEL_PATH)
    return data["model"], data["scaler"], data["features"]

# ============================================================
# PREDICTION
# ============================================================
def predict_fos(user_input):
    model, scaler, features = load_model()

    # 🔥 MAP UI → MODEL
    mapped_input = {}
    for ui_key, model_key in COLUMN_MAP.items():
        if ui_key not in user_input:
            raise KeyError(f"Missing input: {ui_key}")
        mapped_input[model_key] = user_input[ui_key]

    X = np.array([mapped_input[f] for f in features]).reshape(1, -1)
    X_scaled = scaler.transform(X)

    fos = model.predict(X_scaled)[0]
    return round(float(fos), 3)

# ============================================================
# RISK
# ============================================================
def classify_risk(fos):
    if fos < 1:
        return "Danger"
    elif fos < 1.25:
        return "Critical"
    elif fos < 1.5:
        return "Moderate"
    else:
        return "Safe"

# ============================================================
# SENSITIVITY
# ============================================================
def sensitivity_analysis(user_input, variable, num_points=50):
    model, scaler, features = load_model()

    ranges = {
        "Slope Height H (m)": (1, 200),
        "Slope Angle (deg)": (10, 80),
        "Cohesion c (kPa)": (0, 120),
        "Friction Angle (deg)": (10, 45),
        "Unit Weight (kN/m3)": (12, 22),
        "Groundwater": (1, 5)
    }

    x_vals = np.linspace(*ranges[variable], num_points)
    y_vals = []

    for val in x_vals:
        temp_input = user_input.copy()
        temp_input[variable] = val

        fos = predict_fos(temp_input)
        y_vals.append(fos)

    return x_vals.tolist(), y_vals

# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":

    sample_input = {
        "Slope Height H (m)": 50,
        "Slope Angle (deg)": 45,
        "Cohesion c (kPa)": 25,
        "Friction Angle (deg)": 30,
        "Unit Weight (kN/m3)": 18,
        "Groundwater": 3
    }

    fos = predict_fos(sample_input)
    risk = classify_risk(fos)

    print("\nFOS:", fos)
    print("Risk:", risk)

    x, y = sensitivity_analysis(sample_input, "Slope Height H (m)")

    print("\nSensitivity sample:")
    print(x[:5])
    print(y[:5])