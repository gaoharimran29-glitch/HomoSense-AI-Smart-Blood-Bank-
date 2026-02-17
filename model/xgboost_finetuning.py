import matplotlib
matplotlib.use("Agg")  # <- Add this at the very top, before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import uniform, randint
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import json

# =====================================================
# LOAD SINGLE HOSPITAL DATA
# =====================================================
df = pd.read_csv(r"dataset/blood_bank_dataset_final.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["BloodGroup", "Date"])

split_date = "2025-07-01"

# =====================================================
# FEATURE ENGINEERING FUNCTION
# =====================================================
def build_features(data):
    data = data.sort_values("Date").copy()
    data["Lag_1"] = data["UnitsUsed"].shift(1)
    data["Lag_3"] = data["UnitsUsed"].shift(3)
    data["Lag_7"] = data["UnitsUsed"].shift(7)
    data["RollingMean_7"] = data["UnitsUsed"].rolling(7).mean()
    data["DemandVolatility_7"] = data["UnitsUsed"].rolling(7).std()
    data["GrowthRate"] = (data["Lag_1"] - data["Lag_3"]) / (data["Lag_3"] + 1)
    data["ExpiryPressure"] = data["UnitsExpired"] / (data["UnitsAvailable"] + 1)
    data["StockCoverageDays"] = data["UnitsAvailable"] / (data["RollingMean_7"] + 1)
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    return data

tabular_features = [
    "RollingMean_7","Lag_1","Lag_3","Lag_7",
    "GrowthRate","DemandVolatility_7",
    "ExpiryPressure","StockCoverageDays",
    "EmergencyCases","TrafficIndex",
    "AvgTemperature","HolidayFlag",
    "WeekendFlag","DayOfWeek","Month"
]

# =====================================================
# CREATE OUTPUT FOLDER
# =====================================================
output_folder = r"model/Hosp_A_XGB_Results"
os.makedirs(output_folder, exist_ok=True)

# =====================================================
# LOOP PER BLOOD GROUP
# =====================================================
metrics_list = []

for blood in df["BloodGroup"].unique():
    data = df[df["BloodGroup"] == blood].copy()
    train = data[data["Date"] < split_date].copy()
    test = data[data["Date"] >= split_date].copy()
    if len(test) < 30:
        continue

    train = build_features(train).dropna()
    test = build_features(test).dropna()
    
    X_train = train[tabular_features].fillna(0)
    y_train = train["UnitsUsed"]
    X_test = test[tabular_features].fillna(0)
    y_test = test["UnitsUsed"]

    # ------------------- TIME SERIES SPLIT RANDOMIZED SEARCH -------------------
    tscv = TimeSeriesSplit(n_splits=3)
    param_dist = {
        "n_estimators": randint(150, 400),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.2),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4)
    }

    xgb = XGBRegressor(random_state=42, verbosity=0)
    random_search = RandomizedSearchCV(
        xgb, param_distributions=param_dist,
        n_iter=30, cv=tscv, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    pred = best_model.predict(X_test)

    # ------------------- METRICS -------------------
    # ------------------- METRICS -------------------
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)

    metrics_list.append([blood, rmse, mae, r2])

    print(f"✅ {blood} done | RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f}")

    # ------------------- CREATE BLOOD GROUP FOLDER -------------------
    bg_folder = os.path.join(output_folder, blood)
    os.makedirs(bg_folder, exist_ok=True)

    # ------------------- PLOTS -------------------
    # Actual vs Predicted
    plt.figure(figsize=(12,5))
    plt.plot(test["Date"], y_test, label="Actual")
    plt.plot(test["Date"], pred, label="Predicted")
    plt.title(f"Actual vs Predicted - Blood Group {blood}")
    plt.xlabel("Date")
    plt.ylabel("UnitsUsed")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(bg_folder, "actual_vs_predicted.png"))
    plt.close()

    # Feature Importance
    importance = pd.DataFrame({
        "Feature": tabular_features,
        "Importance": best_model.feature_importances_
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(10,6))
    plt.barh(importance["Feature"], importance["Importance"])
    plt.title(f"Feature Importance - Blood Group {blood}")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(bg_folder, "feature_importance.png"))
    plt.close()

    # ------------------- SAVE MODEL -------------------
    import joblib
    joblib.dump(best_model, os.path.join(bg_folder, f"{blood}_xgb_model.pkl"))

    # ------------------- SAVE METRICS -------------------
    metrics_dict = {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2)
    }

    with open(os.path.join(bg_folder, f"{blood}_metrics.json"), "w") as f:
        json.dump(metrics_dict, f, indent=4)

# =====================================================
# FINAL METRICS DISPLAY
# =====================================================
metrics_df = pd.DataFrame(metrics_list, 
                          columns=["BloodGroup","RMSE","MAE","R2"])

metrics_df.to_csv(os.path.join(output_folder, "all_metrics_summary.csv"), index=False)

print("\n📊 Performance Metrics Per Blood Group")
print(metrics_df)

'''
✅ A+ done | RMSE=3.522 MAE=2.781 R2=0.622
✅ A- done | RMSE=2.877 MAE=2.311 R2=0.712
✅ AB+ done | RMSE=1.920 MAE=1.534 R2=0.753
✅ AB- done | RMSE=1.967 MAE=1.415 R2=0.786
✅ B+ done | RMSE=3.969 MAE=3.206 R2=0.675
✅ B- done | RMSE=2.708 MAE=2.174 R2=0.724
✅ O+ done | RMSE=4.116 MAE=3.325 R2=0.694
✅ O- done | RMSE=7.733 MAE=6.121 R2=0.552

📊 Performance Metrics Per Blood Group
  BloodGroup      RMSE       MAE        R2
0         A+  3.521577  2.780813  0.621969
1         A-  2.877367  2.310506  0.712456
2        AB+  1.920121  1.534415  0.753342
3        AB-  1.966765  1.415442  0.786210
4         B+  3.968610  3.205555  0.675343
5         B-  2.707523  2.174330  0.724198
6         O+  4.116065  3.325019  0.694265
7         O-  7.732824  6.121037  0.551506
'''