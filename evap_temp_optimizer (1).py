import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Load and prep data
file=r"/content/single_room_evap_temp_data (1).csv"
df = pd.read_csv(file)
X = df[['setpoint_c', 'humidity_pct', 'outdoor_temp_c', 'occupancy', 'evap_temp_c']]
y = df['compressor_work_kwh']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the tuned model
xgb_model =XGBRegressor(
    n_estimators=500,
    early_stopping_rounds=5,
    learning_rate=0.05,
    n_jobs=4
)
xgb_model.fit(X_train, y_train,
             eval_set=[(X_test, y_test)], 
             verbose=False
              )
predictions=xgb_model.predict(X_test)
print(mean_absolute_error(y_test,predictions))
def get_dew_point(temp, rh):
    a, b = 17.27, 237.3
    alpha = ((a * temp) / (b + temp)) + np.log(rh / 100.0)
    return (b * alpha) / (a - alpha)

def get_safe_te_range(setpoint, humidity, outdoor_temp):
    dew_point = get_dew_point(setpoint, humidity)

    min_te = 2.0
    max_te = dew_point - 1.0

    tc_est = outdoor_temp + 12.0
    if (78.1 - tc_est) < 10.0:
        min_te = max(min_te, 6.0)

    if min_te >= max_te:
        return np.array([min_te])

    return np.arange(min_te, max_te, 0.5)

def optimize_Tev(setpoint, humidity, outdoor, occupancy, model):
    te_options = get_safe_te_range(setpoint, humidity, outdoor)

    test_df = pd.DataFrame({
        'setpoint_c': [setpoint] * len(te_options),
        'humidity_pct': [humidity] * len(te_options),
        'outdoor_temp_c': [outdoor] * len(te_options),
        'occupancy': [occupancy] * len(te_options),
        'evap_temp_c': te_options
    })

    preds = model.predict(test_df)
    best_idx = np.argmin(preds)

    return te_options[best_idx], preds[best_idx]
print(predictions)