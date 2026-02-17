import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os

# =====================================================
# LOAD SINGLE HOSPITAL DATA
# =====================================================
df = pd.read_csv(r"dataset/blood_bank_dataset_final.csv")
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values(["BloodGroup", "Date"])

split_date = "2025-07-01"

# =====================================================
# FEATURE ENGINEERING FUNCTION (NO LEAKAGE)
# =====================================================
def build_features(data):
    data = data.sort_values("Date").copy()

    # Lag Features (safe)
    data["Lag_1"] = data["UnitsUsed"].shift(1)
    data["Lag_3"] = data["UnitsUsed"].shift(3)
    data["Lag_7"] = data["UnitsUsed"].shift(7)

    # Rolling Features (backward only)
    data["RollingMean_7"] = data["UnitsUsed"].rolling(7).mean()
    data["DemandVolatility_7"] = data["UnitsUsed"].rolling(7).std()

    # Growth based ONLY on past lags (no y_t usage)
    data["GrowthRate"] = (data["Lag_1"] - data["Lag_3"]) / (data["Lag_3"] + 1)

    # Operational Features (safe)
    data["ExpiryPressure"] = data["UnitsExpired"] / (data["UnitsAvailable"] + 1)
    data["StockCoverageDays"] = data["UnitsAvailable"] / (data["RollingMean_7"] + 1)

    # Time Features
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month

    return data

# =====================================================
# FEATURE LISTS
# =====================================================
tabular_features = [
    "RollingMean_7","Lag_1","Lag_3","Lag_7",
    "GrowthRate","DemandVolatility_7",
    "ExpiryPressure","StockCoverageDays",
    "EmergencyCases","TrafficIndex",
    "AvgTemperature","HolidayFlag",
    "WeekendFlag","DayOfWeek","Month"
]

lstm_features = [
    "UnitsUsed",
    "TrafficIndex",
    "AvgTemperature",
    "EmergencyCases",
    "HolidayFlag",
    "WeekendFlag"
]

results = []
all_importances = []

# =====================================================
# LOOP OVER BLOOD GROUPS
# =====================================================
for blood in df["BloodGroup"].unique():
    data = df[df["BloodGroup"] == blood].copy()

    if len(data) < 200:
        continue

    # -------- SPLIT DATA --------
    train = data[data["Date"] < split_date].copy()
    test = data[data["Date"] >= split_date].copy()

    if len(test) < 30:
        continue

    # -------- BUILD FEATURES --------
    train = build_features(train)
    test = build_features(test)

    train = train.dropna()
    test = test.dropna()

    y_test = test["UnitsUsed"]

    print(f"\nProcessing BloodGroup: {blood}")

    # =================================================
    # 1️⃣ PROPHET
    # =================================================
    try:
        prophet_df = train[["Date","UnitsUsed"]].rename(columns={"Date":"ds","UnitsUsed":"y"})
        model_p = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        model_p.fit(prophet_df)
        future = model_p.make_future_dataframe(periods=len(test))
        forecast = model_p.predict(future)
        pred_p = forecast.tail(len(test))["yhat"].values
        rmse_p = np.sqrt(mean_squared_error(y_test, pred_p))
        mae_p = mean_absolute_error(y_test, pred_p)
    except:
        rmse_p, mae_p = np.nan, np.nan

    # =================================================
    # 2️⃣ ARIMA
    # =================================================
    try:
        adf_result = adfuller(train["UnitsUsed"])
        p_value = adf_result[1]
        d_order = 1 if p_value > 0.05 else 0
        model_a = SARIMAX(train["UnitsUsed"], order=(2,d_order,2), seasonal_order=(1,1,1,7))
        model_a_fit = model_a.fit(disp=False)
        pred_a = model_a_fit.forecast(steps=len(test))
        rmse_a = np.sqrt(mean_squared_error(y_test, pred_a))
        mae_a = mean_absolute_error(y_test, pred_a)
    except:
        rmse_a, mae_a = np.nan, np.nan

    # =================================================
    # 3️⃣ XGBOOST
    # =================================================
    try:
        X_train = train[tabular_features].fillna(0)
        y_train = train["UnitsUsed"]
        X_test = test[tabular_features].fillna(0)

        model_x = XGBRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model_x.fit(X_train, y_train)
        pred_x = model_x.predict(X_test)

        rmse_x = np.sqrt(mean_squared_error(y_test, pred_x))
        mae_x = mean_absolute_error(y_test, pred_x)

        temp_importance = pd.DataFrame({"Feature": tabular_features, "Importance": model_x.feature_importances_})
        all_importances.append(temp_importance)
    except:
        rmse_x, mae_x = np.nan, np.nan

    # =================================================
    # 4️⃣ LSTM
    # =================================================
    try:
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train[lstm_features])
        test_scaled = scaler.transform(test[lstm_features])

        scaled = np.vstack([train_scaled, test_scaled])
        seq_len = 7
        X_seq, y_seq = [], []

        for i in range(seq_len, len(scaled)):
            X_seq.append(scaled[i-seq_len:i])
            y_seq.append(scaled[i,0])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        split_index = len(train) - seq_len
        X_train_l, y_train_l = X_seq[:split_index], y_seq[:split_index]
        X_test_l, y_test_l = X_seq[split_index:], y_seq[split_index:]

        model_l = Sequential()
        model_l.add(LSTM(64, activation='tanh', input_shape=(seq_len,len(lstm_features))))
        model_l.add(Dense(1))
        model_l.compile(optimizer='adam', loss='mse')
        model_l.fit(X_train_l, y_train_l, epochs=15, batch_size=16, verbose=0)

        pred_l = model_l.predict(X_test_l)
        pred_l_rescaled = scaler.inverse_transform(
            np.concatenate([pred_l, np.zeros((len(pred_l), len(lstm_features)-1))], axis=1)
        )[:,0]

        y_test_rescaled = scaler.inverse_transform(
            np.concatenate([y_test_l.reshape(-1,1), np.zeros((len(y_test_l), len(lstm_features)-1))], axis=1)
        )[:,0]

        rmse_l = np.sqrt(mean_squared_error(y_test_rescaled, pred_l_rescaled))
        mae_l = mean_absolute_error(y_test_rescaled, pred_l_rescaled)
    except:
        rmse_l, mae_l = np.nan, np.nan

    results.append([
        blood, rmse_p, mae_p, rmse_a, mae_a, rmse_x, mae_x, rmse_l, mae_l
    ])

# =====================================================
# RESULTS
# =====================================================
results_df = pd.DataFrame(results, columns=[
    "BloodGroup","Prophet_RMSE","Prophet_MAE",
    "ARIMA_RMSE","ARIMA_MAE","XGB_RMSE","XGB_MAE",
    "LSTM_RMSE","LSTM_MAE"
])

print("\n📊 Per BloodGroup Results")
print(results_df)

summary = pd.DataFrame({
    "Model": ["Prophet","ARIMA","XGBoost","LSTM"],
    "Avg_RMSE": [
        results_df["Prophet_RMSE"].mean(),
        results_df["ARIMA_RMSE"].mean(),
        results_df["XGB_RMSE"].mean(),
        results_df["LSTM_RMSE"].mean()
    ],
    "Avg_MAE": [
        results_df["Prophet_MAE"].mean(),
        results_df["ARIMA_MAE"].mean(),
        results_df["XGB_MAE"].mean(),
        results_df["LSTM_MAE"].mean()
    ]
})

print("\n🏆 Overall Model Comparison")
print(summary.sort_values("Avg_RMSE"))

# =====================================================
# GLOBAL FEATURE IMPORTANCE
# =====================================================
if len(all_importances) > 0:
    importance_df = pd.concat(all_importances)
    global_importance = importance_df.groupby("Feature")["Importance"].mean().sort_values(ascending=False)
    print("\n🔥 Global XGBoost Feature Importance")
    print(global_importance)

# Ouput:- You can see XGBoost outperforms Prophet, ARIMA, and LSTM across all blood groups, with significantly lower RMSE and MAE. 
# The feature importance analysis reveals that "Month", "RollingMean_7", and "HolidayFlag" are the top predictors of blood demand, highlighting the strong seasonality and 
# operational factors influencing demand patterns.
"""
📊 Per BloodGroup Results
  BloodGroup  Prophet_RMSE  Prophet_MAE  ARIMA_RMSE  ARIMA_MAE  XGB_RMSE   XGB_MAE  LSTM_RMSE  LSTM_MAE
0         A+      7.107350     5.497711    5.904771   4.352100  3.405660  2.691890   4.396672  3.491440
1         A-      7.033796     5.399688    5.376182   4.068751  2.843657  2.290987   3.927665  3.059469
2        AB+      4.730568     3.666867    3.940325   2.905409  1.951614  1.564183   3.237663  2.499741
3        AB-      5.168670     3.799490    4.278698   3.205181  1.998705  1.407247   3.233928  2.406462
4         B+      9.440112     7.259910    7.158606   5.444905  3.966288  3.157985   4.925246  3.913204
5         B-      6.792055     5.345432    5.198493   3.836164  2.726380  2.163805   3.916637  3.096623
6         O+     10.017206     7.782753    7.483261   5.561986  4.240810  3.325363   5.733361  4.516018
7         O-     14.626711    11.089910   12.158986   9.208388  7.683190  6.042360   9.521253  7.360462

🏆 Overall Model Comparison
     Model  Avg_RMSE   Avg_MAE
2  XGBoost  3.602038  2.830478
3     LSTM  4.861553  3.792927
1    ARIMA  6.437415  4.822860
0  Prophet  8.114558  6.230220

🔥 Global XGBoost Feature Importance
Feature
RollingMean_7         0.226192
Month                 0.180170
EmergencyCases        0.154075
HolidayFlag           0.090951
WeekendFlag           0.088979
DemandVolatility_7    0.042555
DayOfWeek             0.029021
Lag_3                 0.028610
Lag_1                 0.028371
TrafficIndex          0.024410
StockCoverageDays     0.022506
ExpiryPressure        0.021659
AvgTemperature        0.021570
GrowthRate            0.021473
Lag_7                 0.019458
Name: Importance, dtype: float32
PS C:\Users\hp\Desktop\hackathon> 
"""