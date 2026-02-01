# =========================================================
# Advanced Time Series Forecasting with Prophet
# =========================================================

# -------------------------
# 1. IMPORT LIBRARIES
# -------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error

# -------------------------
# 2. DATA GENERATION
# (Two seasonalities + non-linear trend)
# -------------------------
np.random.seed(42)

dates = pd.date_range(start="2017-01-01", end="2023-12-31", freq="D")
n = len(dates)

# Non-linear trend
trend = 0.02 * np.arange(n) + 0.00002 * (np.arange(n) ** 2)

# Seasonal components
weekly_seasonality = 1.5 * np.sin(2 * np.pi * np.arange(n) / 7)
yearly_seasonality = 3.0 * np.sin(2 * np.pi * np.arange(n) / 365)

# Noise
noise = np.random.normal(0, 0.8, n)

y = 20 + trend + weekly_seasonality + yearly_seasonality + noise

df = pd.DataFrame({
    "ds": dates,
    "y": y
})

# -------------------------
# 3. DATA VISUALIZATION
# -------------------------
plt.figure(figsize=(12,4))
plt.plot(df["ds"], df["y"])
plt.title("Generated Time Series with Weekly and Yearly Seasonality")
plt.xlabel("Date")
plt.ylabel("Value")
plt.show()

# -------------------------
# 4. TRAIN / HOLDOUT SPLIT
# -------------------------
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# -------------------------
# 5. ROLLING WINDOW CV FUNCTION
# -------------------------
def rolling_window_cv(data, params):
    rmses = []

    initial_window = int(len(data) * 0.6)
    horizon = 90
    step = 90

    for i in range(initial_window, len(data) - horizon, step):
        train = data.iloc[:i]
        val = data.iloc[i:i + horizon]

        model = Prophet(**params)
        model.fit(train)

        forecast = model.predict(val)
        rmse = mean_squared_error(val["y"], forecast["yhat"], squared=False)
        rmses.append(rmse)

    return np.mean(rmses)

# -------------------------
# 6. HYPERPARAMETER TUNING
# -------------------------
param_grid = {
    "seasonality_mode": ["additive", "multiplicative"],
    "changepoint_prior_scale": [0.01, 0.05, 0.1, 0.5]
}

results = []

for params in ParameterGrid(param_grid):
    rmse = rolling_window_cv(train_df, params)
    results.append({**params, "rmse": rmse})

results_df = pd.DataFrame(results).sort_values("rmse")
best_params = results_df.iloc[0].to_dict()
best_params.pop("rmse")

print("Best Hyperparameters:")
print(best_params)

# -------------------------
# 7. FINAL MODEL TRAINING
# -------------------------
final_model = Prophet(**best_params)
final_model.fit(train_df)

forecast = final_model.predict(test_df)

# -------------------------
# 8. FORECAST VISUALIZATION
# -------------------------
plt.figure(figsize=(12,4))
plt.plot(train_df["ds"], train_df["y"], label="Train")
plt.plot(test_df["ds"], test_df["y"], label="Test")
plt.plot(test_df["ds"], forecast["yhat"], label="Forecast")
plt.fill_between(
    test_df["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    color="gray",
    alpha=0.3,
    label="Uncertainty Interval"
)
plt.legend()
plt.title("Optimized Prophet Forecast with Uncertainty Intervals")
plt.show()

# -------------------------
# 9. ACCURACY METRICS
# -------------------------
rmse = mean_squared_error(test_df["y"], forecast["yhat"], squared=False)
mae = mean_absolute_error(test_df["y"], forecast["yhat"])

print("RMSE:", rmse)
print("MAE:", mae)

# -------------------------
# 10. UNCERTAINTY CALIBRATION
# -------------------------
def interval_coverage(y_true, lower, upper):
    return np.mean((y_true >= lower) & (y_true <= upper))

coverage_80 = interval_coverage(
    test_df["y"],
    forecast["yhat_lower"],
    forecast["yhat_upper"]
)

print("Empirical 80% Interval Coverage:", coverage_80)

# -------------------------
# 11. FINAL SUMMARY OUTPUT
# -------------------------
summary = {
    "RMSE": rmse,
    "MAE": mae,
    "Empirical_80pct_Coverage": coverage_80,
    "Best_Parameters": best_params
}

summary
