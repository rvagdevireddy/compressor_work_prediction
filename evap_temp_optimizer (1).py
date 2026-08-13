import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# ---------------------------------------------------------
# 1. LOAD DATA & TRAIN THE ADVANCED ML MODEL
# ---------------------------------------------------------
# Load the historical dataset
data = pd.read_csv('single_room_evap_temp_data.csv')
features = ['setpoint_c', 'humidity_pct', 'outdoor_temp_c', 'occupancy', 'evap_temp_c']
target = 'compressor_work_kwh'

X = data[features]
y = data[target]

# Split the data to verify model accuracy on unseen conditions
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Advanced Random Forest (Tuned Hyperparameters)
print("Training Advanced Random Forest (500 Trees, Full Sensor Awareness)...")
rf_advanced = RandomForestRegressor(
    n_estimators=500, 
    max_depth=None, 
    min_samples_split=2, 
    max_features=1.0, 
    random_state=42, 
    n_jobs=-1
)
rf_advanced.fit(X_train, y_train)

# Verify precision mathematically
predictions = rf_advanced.predict(X_valid)
mae = mean_absolute_error(y_valid, predictions)
print(f"Model successfully trained. Mean Absolute Error: {mae:.5f} kWh\n")

# ---------------------------------------------------------
# 2. THE PHYSICS ENGINE (Thermodynamic Guardrails)
# ---------------------------------------------------------
def calculate_dew_point(temp_c, rh_percent):
    """Calculates Dew Point using the Magnus-Tetens formula."""
    a, b = 17.27, 237.3
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh_percent / 100.0)
    return (b * alpha) / (a - alpha)

def get_valid_Te_range(setpoint, humidity, outdoor_temp):
    """Calculates the safe, physically possible Evaporation Temperature limits."""
    dew_point = calculate_dew_point(setpoint, humidity)
    
    # Floor: Must stay safely above freezing (2.0 C)
    min_Te = 2.0 
    
    # Ceiling: Must stay below Dew Point to dehumidify (with a 1C buffer)
    max_Te = dew_point - 1.0 
    
    # Condensing Temp Sanity Check (Assuming R32 Refrigerant)
    Tc_estimated = outdoor_temp + 12.0
    T_crit_R32 = 78.1
    if (T_crit_R32 - Tc_estimated) < 10.0:
        min_Te = max(min_Te, 6.0) # Force higher T_e to protect the compressor

    if min_Te >= max_Te:
        return np.array([min_Te]) # Fallback if boundaries ever cross
        
    # Return array of testable targets, stepping by 0.5 degrees
    return np.arange(min_Te, max_Te, 0.5) 

# ---------------------------------------------------------
# 3. THE OPTIMIZER (The "What-If" Loop)
# ---------------------------------------------------------
def optimize_expansion_valve(setpoint, humidity, outdoor_temp, occupancy, trained_model):
    """Finds the most energy-efficient T_e within the safe physics boundaries."""
    
    # Step 1: Get the safe boundaries from the Physics Engine
    valid_Te_options = get_valid_Te_range(setpoint, humidity, outdoor_temp)
    
    # Step 2: Build the "What-If" simulation scenarios
    scenarios = pd.DataFrame({
        'setpoint_c': [setpoint] * len(valid_Te_options),
        'humidity_pct': [humidity] * len(valid_Te_options),
        'outdoor_temp_c': [outdoor_temp] * len(valid_Te_options),
        'occupancy': [occupancy] * len(valid_Te_options),
        'evap_temp_c': valid_Te_options
    })
    
    # Step 3: Use the Advanced ML Model to predict power for all scenarios instantly
    predicted_work = trained_model.predict(scenarios)
    
    # Step 4: Pick the T_e that resulted in the lowest power consumption
    best_idx = np.argmin(predicted_work)
    
    return valid_Te_options[best_idx], predicted_work[best_idx], valid_Te_options

# ---------------------------------------------------------
# 4. LIVE SYSTEM RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    # Simulated live sensors reading from the room
    live_setpoint = 24.0
    live_humidity = 72.0
    live_outdoor = 34.5
    live_occupancy = 3

    # Execute the optimization loop
    optimal_Te, min_work, valid_range = optimize_expansion_valve(
        live_setpoint, live_humidity, live_outdoor, live_occupancy, rf_advanced
    )

    # Output the final command
    print("--- LIVE SYSTEM OPTIMIZATION ---")
    print(f"Sensor Readings: {live_outdoor}°C Outdoor | {live_humidity}% RH | {live_occupancy} Occupants | {live_setpoint}°C Setpoint")
    print(f"Physics Guardrails: Testing safe T_e states from {valid_range[0]:.1f}°C to {valid_range[-1]:.1f}°C")
    print(f">> COMMAND EEV TARGET (Optimal T_e) : {optimal_Te:.1f}°C")
    print(f">> PREDICTED MINIMUM COMPRESSOR WORK: {min_work:.4f} kWh")
