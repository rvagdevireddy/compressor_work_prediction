"""
Evaporator temperature optimizer for a single-room domestic AC.

Goal: given a desired room setpoint and ambient conditions, find the
evaporating temperature (T_evap) that meets the required cooling load
with the LEAST compressor work, subject to two physical constraints:

  1. FROST CONSTRAINT (evaporator side)
     Frost forms on the coil only when it is BOTH below freezing AND
     below the dew point of the air passing over it (i.e. there is
     liquid moisture available to freeze). If T_evap is below freezing
     but ABOVE the dew point, no condensation happens at all, so there
     is nothing to freeze. We compute dew point from room temp + RH
     (Magnus-Tetens) and clamp T_evap so frost never forms.

  2. CRITICAL POINT CONSTRAINT (condenser side)
     The condensing temperature must stay below the refrigerant's
     critical temperature, or it physically cannot condense (the cycle
     breaks down / goes supercritical). T_cond is solved from a heat
     balance: Q_cond = Q_evap + W_compressor = UA_cond * (T_cond - T_outdoor),
     iterated because W depends on T_cond through COP. If the solved
     T_cond would exceed the critical point, that operating condition
     is flagged infeasible.

Why the lowest-work T_evap is simply "as warm as possible": COP rises
monotonically with T_evap (smaller lift = less compressor work), so the
optimal strategy is always to run the WARMEST evaporator temperature that
still satisfies the load -- i.e. solve the energy balance directly rather
than searching. The frost limit is the only thing that can force T_evap
colder than that ideal (never warmer), so minimal work = the analytic
T_evap, clipped only if frost forces it up.

Output: a CSV of (inputs -> T_evap, T_cond, COP, work) rows you can train
an ML model on, either to predict T_evap directly or to learn the whole
input -> work mapping.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------
# Tunable system parameters -- edit to match your real unit/room
# ---------------------------------------------------------------------
ROOM_PARAMS = {
    "insulation_quality": 0.7,     # 0 (poor) - 1 (excellent)
    "room_area_m2": 15,
    "base_climate_temp": 27,       # avg outdoor temp for your city (C)
    "climate_seasonal_swing": 8,   # +/- deg C across the year
    "climate_daily_swing": 5,      # +/- deg C across a day
}

UA_EVAP_KW_PER_C = 0.07   # evaporator coil heat-transfer conductance
UA_COND_KW_PER_C = 0.12   # condenser coil heat-transfer conductance
COMPRESSOR_ETA = 0.50      # fraction of Carnot COP a real compressor achieves

FREEZE_POINT_C = 0.0
FROST_MARGIN_C = 1.0       # keep coil >= freeze point + margin when frost risk exists
MAX_REALISTIC_COP = 8.0    # practical ceiling (cycling losses, motor/drive inefficiency)

REFRIGERANTS = {
    "R32":   {"critical_temp_c": 78.11},
    "R410A": {"critical_temp_c": 71.36},
    "R134a": {"critical_temp_c": 101.06},
}
REFRIGERANT = "R32"

COMFORT_MIN, COMFORT_MAX = 20, 26
DAYS = 365
OCCUPANCY_MIN, OCCUPANCY_MAX = 1, 4   # occupants per row: never 0, never more than 4
NIGHT_HOURS = [22, 23, 0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------
# Weather / load model (same room physics as before)
# ---------------------------------------------------------------------
def outdoor_temp(day, hour_of_day):
    seasonal = ROOM_PARAMS["climate_seasonal_swing"] * np.sin(2 * np.pi * day / 365)
    daily = ROOM_PARAMS["climate_daily_swing"] * np.sin(2 * np.pi * (hour_of_day - 9) / 24)
    return ROOM_PARAMS["base_climate_temp"] + seasonal + daily + np.random.normal(0, 0.6)


def cooling_load_kw(out_temp_c, setpoint_c, occupancy):
    """Required heat removal rate (kW) to hold setpoint against outdoor heat gain."""
    insulation = ROOM_PARAMS["insulation_quality"]
    gap = max(0.0, out_temp_c - setpoint_c)  # only cooling load, not heating
    leak_factor = 1.3 - insulation
    base_load = gap * leak_factor * ROOM_PARAMS["room_area_m2"] * 0.03
    occupancy_gain = occupancy * 0.15
    return max(0.05, base_load + occupancy_gain)


def dew_point_c(temp_c, rh_pct):
    """Magnus-Tetens approximation."""
    a, b = 17.62, 243.12
    rh = np.clip(rh_pct, 1, 100) / 100.0
    gamma = (a * temp_c) / (b + temp_c) + np.log(rh)
    return (b * gamma) / (a - gamma)


# ---------------------------------------------------------------------
# Core cycle solver
# ---------------------------------------------------------------------
def solve_evap_temp(setpoint_c, load_kw, dew_pt_c):
    """
    Analytic evap temp needed to deliver `load_kw` while holding the
    room at `setpoint_c`, then apply the frost constraint.
    Returns (evap_temp_c, frost_limited: bool).
    """
    ideal_evap = setpoint_c - load_kw / UA_EVAP_KW_PER_C

    frost_would_form = (ideal_evap < FREEZE_POINT_C) and (ideal_evap < dew_pt_c)
    if frost_would_form:
        floor = max(FREEZE_POINT_C + FROST_MARGIN_C, dew_pt_c)
        # if dew point itself is below freezing there's no moisture to
        # freeze, so only the freeze-point+margin floor applies
        floor = FREEZE_POINT_C + FROST_MARGIN_C if dew_pt_c < FREEZE_POINT_C else floor
        return floor, True
    return ideal_evap, False


def solve_condensing_cycle(evap_temp_c, load_kw, outdoor_temp_c, iterations=30):
    """
    Iteratively solve T_cond from the condenser heat balance
    Q_cond = Q_evap + W_compressor, then check against the refrigerant's
    critical temperature.
    Returns (cond_temp_c, cop, work_kw, critical_point_exceeded: bool).
    """
    t_cond = outdoor_temp_c + 10.0  # initial guess (approach temp)
    cop, work_kw = None, None

    for _ in range(iterations):
        t_evap_k = evap_temp_c + 273.15
        t_cond_k = t_cond + 273.15
        delta_t = max(t_cond_k - t_evap_k, 0.5)  # avoid divide-by-zero
        cop = min(COMPRESSOR_ETA * (t_evap_k / delta_t), MAX_REALISTIC_COP)
        work_kw = load_kw / cop
        q_cond_kw = load_kw + work_kw
        t_cond_new = outdoor_temp_c + q_cond_kw / UA_COND_KW_PER_C
        t_cond = 0.5 * t_cond + 0.5 * t_cond_new  # damped update for stability

    critical_temp = REFRIGERANTS[REFRIGERANT]["critical_temp_c"]
    exceeded = t_cond >= critical_temp
    if exceeded:
        # infeasible operating point -- cap at just below critical and
        # recompute so downstream numbers stay finite (row gets flagged)
        t_cond = critical_temp - 0.5
        t_evap_k = evap_temp_c + 273.15
        t_cond_k = t_cond + 273.15
        cop = min(COMPRESSOR_ETA * (t_evap_k / max(t_cond_k - t_evap_k, 0.5)), MAX_REALISTIC_COP)
        work_kw = load_kw / cop

    return t_cond, cop, work_kw, exceeded


def solve_row(outdoor_temp_c, setpoint_c, rh_pct, occupancy):
    load_kw = cooling_load_kw(outdoor_temp_c, setpoint_c, occupancy)
    dew_pt = dew_point_c(setpoint_c, rh_pct)  # dew point of room air, not outdoor
    evap_temp_c, frost_limited = solve_evap_temp(setpoint_c, load_kw, dew_pt)
    cond_temp_c, cop, work_kw, crit_exceeded = solve_condensing_cycle(
        evap_temp_c, load_kw, outdoor_temp_c
    )
    return {
        "load_kw": round(load_kw, 4),
        "dew_point_c": round(dew_pt, 2),
        "evap_temp_c": round(evap_temp_c, 2),
        "frost_limited": frost_limited,
        "cond_temp_c": round(cond_temp_c, 2),
        "critical_point_exceeded": crit_exceeded,
        "cop": round(cop, 3),
        "compressor_work_kwh": round(work_kw, 4),  # 1-hour timestep -> kWh == kW
    }


# ---------------------------------------------------------------------
# Dataset generation (night hours only, matching the earlier dataset)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    rows = []
    for day in range(DAYS):
        for hour_of_day in NIGHT_HOURS:
            out_t = outdoor_temp(day, hour_of_day)
            setpoint_c = round(np.clip(np.random.normal(23, 1.5), COMFORT_MIN, COMFORT_MAX), 1)
            rh_pct = round(np.clip(np.random.normal(55, 12), 20, 95), 1)
            occupancy = int(np.random.randint(OCCUPANCY_MIN, OCCUPANCY_MAX + 1))

            result = solve_row(out_t, setpoint_c, rh_pct, occupancy)

            rows.append({
                "day": day,
                "hour_of_day": hour_of_day,
                "outdoor_temp_c": round(out_t, 2),
                "setpoint_c": setpoint_c,
                "humidity_pct": rh_pct,
                "occupancy": occupancy,
                "refrigerant": REFRIGERANT,
                **result,
            })

    df = pd.DataFrame(rows)
    out_path = "/mnt/user-data/outputs/single_room_evap_temp_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Frost-limited rows: {df['frost_limited'].sum()}")
    print(f"Critical-point-exceeded rows: {df['critical_point_exceeded'].sum()}")
    print(df.head(10).to_string())
