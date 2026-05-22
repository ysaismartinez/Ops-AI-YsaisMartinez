"""
Precomputes all demand aggregations from demand_enriched.parquet at startup.
Keeps only ~44K rows in memory (zone × hour × dow profile), not 6.3M raw rows.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import requests
from functools import lru_cache

_ROOT = Path("/")
DATA_PATH = _ROOT / "data" / "processed" / "demand_enriched.parquet"
LOOKUP_PATH = _ROOT / "metadata" / "Lookups" / "taxi_zone_lookup.csv"
MODEL_PATH = _ROOT / "data" / "processed" / "lgbm_demand_model.txt"

# Fixed reference point: end of 2nd week in Feb 2026 (the latest complete month)
# Data before this date is actual; from this point forward uses model predictions
REFERENCE_DATE = pd.Timestamp("2026-02-14")

AIRPORT_ZONES = {1, 132, 138}  # Newark, JFK, LaGuardia

# Holiday definitions (month, day)
HOLIDAYS = {
    (1, 1): "New Year's Day",
    (1, 20): "MLK Day",
    (2, 17): "Presidents Day",
    (3, 17): "St. Patrick's Day",
    (5, 26): "Memorial Day",
    (7, 4): "Independence Day",
    (9, 1): "Labor Day",
    (10, 13): "Columbus Day",
    (10, 31): "Halloween",
    (11, 11): "Veterans Day",
    (11, 27): "Thanksgiving",  # 4th Thu of Nov - approximate
    (12, 24): "Christmas Eve",
    (12, 25): "Christmas",
    (12, 31): "New Year's Eve",
}


def _identify_holiday(date: pd.Timestamp) -> str:
    """Identify holiday name from date, or return 'regular'."""
    key = (date.month, date.day)
    return HOLIDAYS.get(key, "regular")


def _get_week_context(date_str: str) -> tuple:
    """
    Get date context for weighting historical data.
    Returns (date, month, day, is_holiday).
    Dates near holidays should weight historical data more carefully.
    """
    target_date = pd.Timestamp(date_str)
    holiday = _identify_holiday(target_date)
    return target_date, target_date.month, target_date.day, holiday


# Features for LightGBM model
FEATURES = [
    "PULocationID",
    "hour",
    "minute",
    "dayofweek",
    "is_weekend",
    "month",
    "dayofyear",
    "weekofyear",
    "year",
    "slot_of_day",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "is_holiday",
    "cbd_pricing_active",
    "is_airport_zone",
    "borough_id",
    "service_zone_id",
    "zone_slot_baseline",
    "lag_15min",
    "lag_1h",
    "lag_2h",
    "lag_1day",
    "lag_1week",
    "roll_mean_1h",
    "roll_mean_2h",
    "roll_mean_1day",
]


def _load():
    print("[NYC Cab Analytics] Loading demand profile...")
    df = pd.read_parquet(
        DATA_PATH,
        columns=[
            "PULocationID",
            "hour",
            "dayofweek",
            "trip_count",
            "is_holiday",
            "time_bucket",
        ],
    )

    # Identify specific holidays
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    df["holiday_name"] = df.apply(
        lambda row: (
            _identify_holiday(row["time_bucket"])
            if row["is_holiday"] == 1
            else "regular"
        ),
        axis=1,
    )

    # Profile 1: Regular days (non-holidays)
    regular_df = df[df["is_holiday"] == 0].drop(
        columns=["is_holiday", "time_bucket", "holiday_name"]
    )
    regular_profile = (
        regular_df.groupby(["PULocationID", "hour", "dayofweek"], as_index=False)[
            "trip_count"
        ]
        .mean()
        .rename(columns={"trip_count": "avg"})
    )
    regular_profile["avg"] = regular_profile["avg"].round(3)
    regular_profile["holiday_name"] = "regular"

    # Profile 2: Holiday-specific demand
    holiday_df = df[df["is_holiday"] == 1].copy()
    holiday_profile = (
        holiday_df.groupby(["PULocationID", "hour", "holiday_name"], as_index=False)[
            "trip_count"
        ]
        .mean()
        .rename(columns={"trip_count": "avg"})
    )
    holiday_profile["avg"] = holiday_profile["avg"].round(3)
    # Add dayofweek as -1 for holidays (since it varies)
    holiday_profile["dayofweek"] = -1

    # Combine both profiles
    profile = pd.concat([regular_profile, holiday_profile], ignore_index=True)

    zones_df = pd.read_csv(LOOKUP_PATH).rename(
        columns={
            "LocationID": "zone_id",
            "Zone": "name",
            "Borough": "borough",
            "service_zone": "service_zone",
        }
    )
    print(
        f"[NYC Cab Analytics] Profile ready — {len(profile):,} rows, {profile['PULocationID'].nunique()} zones"
    )
    print(f"[NYC Cab Analytics]   Regular days: {len(regular_profile):,} rows")
    print(f"[NYC Cab Analytics]   Holidays: {len(holiday_profile):,} rows")
    return profile, zones_df


def _load_model():
    try:
        import lightgbm as lgb

        model = lgb.Booster(model_file=str(MODEL_PATH))
        print("[NYC Cab Analytics] LightGBM model loaded.")
        return model
    except Exception as e:
        print(f"[NYC Cab Analytics] Error loading LightGBM model: {e}")
        return None


def _load_full_demand():
    """Load full demand data for lag calculation."""
    print("[NYC Cab Analytics] Loading full demand data for forecasting...")
    df = pd.read_parquet(DATA_PATH)
    df["time_bucket"] = pd.to_datetime(df["time_bucket"])
    return df


_profile, _zones_df = _load()
print("[DEBUG] _zones_df columns:", _zones_df.columns.tolist())
print("[DEBUG] _zones_df head:", _zones_df.head().to_dict("records"))
zone_key = "zone_id" if "zone_id" in _zones_df.columns else "LocationID"
_zone_map = _zones_df.set_index(zone_key).to_dict("index")
_lgbm_model = _load_model()
_full_demand = _load_full_demand() if _lgbm_model else None


# ── Zone-Hour Average Fares (for realistic earnings estimates) ─────────────────


def _load_zone_hour_fares():
    """Load average fare per zone-hour from preprocessed data."""
    try:
        fare_path = _ROOT / "data" / "processed" / "zone_hour_avg_fare.parquet"
        df = pd.read_parquet(fare_path)
        # Create lookup: {(zone_id, hour): avg_fare}
        return dict(zip(zip(df["zone_id"], df["hour"]), df["avg_fare"]))
    except Exception as e:
        print(f"[NYC Cab Analytics] Warning: Could not load zone fares: {e}")
        # Fallback to overall average
        return {}


_zone_hour_fares = _load_zone_hour_fares()
_fallback_avg_fare = 18.70  # Overall average from actual data


def _get_zone_hour_fare(zone_id: int, hour: int) -> float:
    """Get average fare for a zone at a specific hour."""
    fare = _zone_hour_fares.get((zone_id, hour))
    if fare is None:
        fare = _fallback_avg_fare
    return max(5.0, min(100.0, fare))  # Clamp to reasonable range


# ── Zone Coordinates (for real drive time calculation) ────────────────────────


def _load_zone_coordinates():
    """Load zone centroids from geojson for OSRM routing."""
    try:
        geojson_path = _ROOT / "app" / "frontend" / "public" / "taxi_zones.geojson"
        with open(geojson_path) as f:
            geojson = json.load(f)

        zone_coords = {}
        for feature in geojson["features"]:
            zone_id = feature["properties"].get("LocationID")
            geom = feature["geometry"]

            if geom["type"] == "Polygon":
                coords = geom["coordinates"][0]
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                zone_coords[int(zone_id)] = {
                    "lat": (min(lats) + max(lats)) / 2,
                    "lon": (min(lons) + max(lons)) / 2,
                }

        return zone_coords
    except Exception as e:
        print(f"[NYC Cab Analytics] Warning: Could not load zone coordinates: {e}")
        return {}


_zone_coords = _load_zone_coordinates()


# ── Drive Time Calculation (OSRM - Open Source Routing Machine) ──────────────


@lru_cache(maxsize=500)
def _get_drive_time(from_zone: int, to_zone: int) -> int:
    """
    Calculate real driving time between zones using OSRM.
    Falls back to distance-based estimate if OSRM fails.
    Returns drive time in minutes.
    """
    if from_zone not in _zone_coords or to_zone not in _zone_coords:
        # Fallback: estimate 3 mins per mile, ~2 miles avg per zone
        return max(3, abs(from_zone - to_zone) // 20)  # rough heuristic

    try:
        from_coord = _zone_coords[from_zone]
        to_coord = _zone_coords[to_zone]

        # OSRM public API endpoint (free, open source)
        url = f"http://router.project-osrm.org/route/v1/driving/{from_coord['lon']},{from_coord['lat']};{to_coord['lon']},{to_coord['lat']}?overview=false"

        response = requests.get(url, timeout=3)
        if response.status_code == 200:
            data = response.json()
            if data["routes"]:
                # Duration is in seconds, convert to minutes
                duration_mins = int(data["routes"][0]["duration"] / 60)
                return max(2, duration_mins)  # minimum 2 mins

    except Exception as e:
        pass  # Fall through to fallback

    # Fallback: rough distance estimate
    from_coord = _zone_coords.get(from_zone, {"lat": 40.75, "lon": -73.97})
    to_coord = _zone_coords.get(to_zone, {"lat": 40.75, "lon": -73.97})

    # Simple distance estimate (degrees * 69 miles per degree)
    lat_diff = abs(from_coord["lat"] - to_coord["lat"]) * 69
    lon_diff = abs(from_coord["lon"] - to_coord["lon"]) * 54
    distance_miles = (lat_diff**2 + lon_diff**2) ** 0.5

    # Assume 15 mph average in NYC traffic
    drive_time = max(2, int(distance_miles / 15 * 60))
    return min(45, drive_time)  # cap at 45 mins


# ── Volatility Cache (Confidence/Stability Signals) ──────────────────────────


def _compute_zone_volatility():
    """
    Compute historical demand volatility (std dev) for each (zone, hour) pair.
    Used to flag high-volatility zones where surge pricing is risky.
    Returns dict: (zone_id, hour) -> volatility_ratio (0-1, higher = more volatile)
    """
    volatility = {}

    if _full_demand is None or _full_demand.empty:
        return volatility

    for zone_id in _profile["PULocationID"].unique():
        for hour in range(24):
            hour_data = _full_demand[
                (_full_demand["PULocationID"] == zone_id)
                & (_full_demand["hour"] == hour)
            ]["trip_count"]

            if len(hour_data) > 1:
                mean_demand = hour_data.mean()
                std_demand = hour_data.std()
                # Volatility ratio: std / mean (0-1, higher = more volatile)
                ratio = (std_demand / mean_demand) if mean_demand > 0 else 0
                volatility[(zone_id, hour)] = ratio

    return volatility


_zone_volatility = _compute_zone_volatility()


# ── Unmet Demand Baselines (Data-Driven) ────────────────────────────────────


def _compute_zone_unmet_demand_baseline():
    """
    Learn from actual data: what demand levels indicate stress?
    Uses the 75th percentile of historical demand per zone (from profile).
    Demand above this threshold indicates the zone is busier than "normal".

    Returns dict: zone_id -> 75th percentile demand value (threshold for "tight")
    """
    unmet_baselines = {}

    if _profile is None or _profile.empty:
        return unmet_baselines

    for zone_id in _profile["PULocationID"].unique():
        zone_profile_demand = _profile[_profile["PULocationID"] == zone_id]["avg"]

        if len(zone_profile_demand) > 0:
            # Get 75th percentile - represents "normal busy" demand
            # Anything above this suggests understaffing/unmet demand
            p75 = zone_profile_demand.quantile(0.75)
            unmet_baselines[zone_id] = p75

    return unmet_baselines


_zone_unmet_baselines = _compute_zone_unmet_demand_baseline()


# ── Synthetic Live Data Generation ──────────────────────────────────────────


def _generate_synthetic_current_demand(hour: int, dow: int) -> dict:
    """
    Generate synthetic 'live' demand for current moment.
    Uses historical baseline + intelligent variations (seasonality, noise, events).

    Args:
        hour: Current hour (0-23)
        dow: Current day of week (0-6)

    Returns:
        Dict with zone_id -> synthetic demand value
    """
    # Get historical baseline for this hour/dow
    baseline = _profile[
        (_profile["hour"] == hour) & (_profile["dayofweek"] == dow)
    ].copy()

    synthetic = {}
    for _, row in baseline.iterrows():
        zone_id = int(row["PULocationID"])
        base_demand = float(row["avg"])

        # Intelligent variations:
        # 1. Random walk (±10% variation)
        noise_factor = np.random.normal(1.0, 0.08)

        # 2. Rush hour boost (8-10 AM, 5-7 PM)
        rush_hour_boost = 1.2 if hour in [8, 9, 17, 18] else 1.0

        # 3. Weather-like pattern (slight mid-day dip, evening peak)
        time_pattern = 1.0 + 0.1 * np.sin(2 * np.pi * (hour - 6) / 24)

        # 4. Zone-specific variation
        zone_variance = 1.0 + (np.random.random() - 0.5) * 0.15

        synthetic_demand = (
            base_demand * noise_factor * rush_hour_boost * time_pattern * zone_variance
        )
        synthetic_demand = max(0.0, synthetic_demand)  # No negative demand

        synthetic[zone_id] = round(synthetic_demand, 2)

    return synthetic


def get_synthetic_current_demand(hour: int, dow: int, date: str = None) -> dict:
    """API endpoint: get synthetic 'live' demand for any hour/dow."""
    synthetic = _generate_synthetic_current_demand(hour, dow)
    max_val = max(synthetic.values(), default=1.0)

    return {
        "demand": {str(k): v for k, v in synthetic.items()},
        "max": round(max_val, 2),
        "hour": hour,
        "dow": dow,
        "is_synthetic": True,
        "label": "Synthetic Live Data",
    }


# ── LightGBM Forecasting ─────────────────────────────────────────────────────


def _compute_temporal_features(time_bucket: pd.Timestamp) -> dict:
    """Compute temporal, cyclical, and derived features for a given timestamp."""
    hour = time_bucket.hour
    minute = time_bucket.minute
    dow = time_bucket.dayofweek
    month = time_bucket.month
    day_of_year = time_bucket.dayofyear
    week_of_year = time_bucket.isocalendar().week
    year = time_bucket.year
    slot_of_day = hour * 4 + minute // 15  # 0-95 slots per day

    # Cyclical encoding
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    is_weekend = 1 if dow in [5, 6] else 0

    return {
        "hour": hour,
        "minute": minute,
        "dayofweek": dow,
        "is_weekend": is_weekend,
        "month": month,
        "dayofyear": day_of_year,
        "weekofyear": week_of_year,
        "year": year,
        "slot_of_day": slot_of_day,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    }


def _get_zone_history(zone_id: int, num_slots: int = 672) -> pd.DataFrame:
    """Get recent historical data for a zone (default: 7 days = 672 x 15-min slots)."""
    if _full_demand is None:
        return pd.DataFrame()
    zone_data = _full_demand[_full_demand["PULocationID"] == zone_id].copy()
    return zone_data.tail(num_slots).sort_values("time_bucket").reset_index(drop=True)


def _calculate_lags_and_rolling(history: pd.DataFrame, last_time: pd.Timestamp) -> dict:
    """Calculate lag and rolling mean features based on historical data."""
    if history.empty:
        return {
            "lag_15min": np.nan,
            "lag_1h": np.nan,
            "lag_2h": np.nan,
            "lag_1day": np.nan,
            "lag_1week": np.nan,
            "roll_mean_1h": np.nan,
            "roll_mean_2h": np.nan,
            "roll_mean_1day": np.nan,
        }

    # Get values at specific lag points (looking backwards from last_time)
    t_minus_15min = last_time - timedelta(minutes=15)
    t_minus_1h = last_time - timedelta(hours=1)
    t_minus_2h = last_time - timedelta(hours=2)
    t_minus_1day = last_time - timedelta(days=1)
    t_minus_1week = last_time - timedelta(days=7)

    def get_trip_count_at(time_point):
        matches = history[history["time_bucket"] == time_point]
        return float(matches["trip_count"].values[0]) if len(matches) > 0 else np.nan

    lag_15min = get_trip_count_at(t_minus_15min)
    lag_1h = get_trip_count_at(t_minus_1h)
    lag_2h = get_trip_count_at(t_minus_2h)
    lag_1day = get_trip_count_at(t_minus_1day)
    lag_1week = get_trip_count_at(t_minus_1week)

    # Rolling means over last hour, 2 hours, 1 day
    recent_1h = history[history["time_bucket"] > last_time - timedelta(hours=1)]
    recent_2h = history[history["time_bucket"] > last_time - timedelta(hours=2)]
    recent_1day = history[history["time_bucket"] > last_time - timedelta(days=1)]

    roll_mean_1h = recent_1h["trip_count"].mean() if len(recent_1h) > 0 else np.nan
    roll_mean_2h = recent_2h["trip_count"].mean() if len(recent_2h) > 0 else np.nan
    roll_mean_1day = (
        recent_1day["trip_count"].mean() if len(recent_1day) > 0 else np.nan
    )

    return {
        "lag_15min": lag_15min,
        "lag_1h": lag_1h,
        "lag_2h": lag_2h,
        "lag_1day": lag_1day,
        "lag_1week": lag_1week,
        "roll_mean_1h": roll_mean_1h,
        "roll_mean_2h": roll_mean_2h,
        "roll_mean_1day": roll_mean_1day,
    }


def forecast_demand(
    zone_id: int, hour: int, dow: int, num_steps: int = 16, date: str = None
) -> list:
    """
    Forecast demand for a zone using LightGBM with synthetic lags.
    Uses synthetic current demand as the baseline for realistic lag calculations.

    Args:
        zone_id: Zone ID to forecast for
        hour: Current hour
        dow: Current day of week
        num_steps: Number of 15-min steps to forecast (default 16 = 4 hours)

    Returns:
        List of forecast dicts with time_bucket and predicted demand
    """
    if _lgbm_model is None or _full_demand is None:
        return []

    # Get zone metadata
    zone_info = _zone_map.get(zone_id, {})

    # Get synthetic current demand to use as baseline for lags
    synthetic_current = _generate_synthetic_current_demand(hour, dow)
    current_synthetic_demand = synthetic_current.get(zone_id, 0.0)

    # Build synthetic history starting from current time going back
    now = pd.Timestamp.now()
    synthetic_history = []

    # Add 7 days of synthetic history (backwards from now)
    for days_back in range(7, 0, -1):
        for hr in range(24):
            for slot in range(4):  # 4 x 15-min slots per hour
                past_time = now - timedelta(days=days_back, hours=hr, minutes=slot * 15)
                past_hour = past_time.hour
                past_dow = past_time.dayofweek

                past_synthetic = _generate_synthetic_current_demand(past_hour, past_dow)
                past_demand = past_synthetic.get(zone_id, 0.0)

                synthetic_history.append(
                    {
                        "time_bucket": past_time,
                        "trip_count": past_demand,
                        "borough_id": (
                            int(
                                _full_demand[_full_demand["PULocationID"] == zone_id][
                                    "borough_id"
                                ].iloc[0]
                            )
                            if len(
                                _full_demand[_full_demand["PULocationID"] == zone_id]
                            )
                            > 0
                            else 0
                        ),
                        "service_zone_id": (
                            int(
                                _full_demand[_full_demand["PULocationID"] == zone_id][
                                    "service_zone_id"
                                ].iloc[0]
                            )
                            if len(
                                _full_demand[_full_demand["PULocationID"] == zone_id]
                            )
                            > 0
                            else 0
                        ),
                        "is_airport_zone": 1 if zone_id in AIRPORT_ZONES else 0,
                        "zone_slot_baseline": (
                            float(
                                _full_demand[_full_demand["PULocationID"] == zone_id][
                                    "zone_slot_baseline"
                                ].iloc[0]
                            )
                            if len(
                                _full_demand[_full_demand["PULocationID"] == zone_id]
                            )
                            > 0
                            else 0.0
                        ),
                    }
                )

    synthetic_history_df = pd.DataFrame(synthetic_history)

    # Get zone metadata from actual data
    zone_data = _full_demand[_full_demand["PULocationID"] == zone_id]
    if zone_data.empty:
        return []

    last_borough_id = int(zone_data["borough_id"].iloc[0])
    last_service_zone_id = int(zone_data["service_zone_id"].iloc[0])
    is_airport = int(zone_data["is_airport_zone"].iloc[0])
    zone_slot_baseline = float(zone_data["zone_slot_baseline"].iloc[0])
    is_holiday = 0
    cbd_pricing_active = 0

    # Add current synthetic demand to history
    synthetic_history_df = pd.concat(
        [
            synthetic_history_df,
            pd.DataFrame(
                {
                    "time_bucket": [now],
                    "trip_count": [current_synthetic_demand],
                    "borough_id": [last_borough_id],
                    "service_zone_id": [last_service_zone_id],
                    "is_airport_zone": [is_airport],
                    "zone_slot_baseline": [zone_slot_baseline],
                }
            ),
        ],
        ignore_index=True,
    )

    # Start predictions
    predictions = []
    current_time = now + timedelta(minutes=15)

    for step in range(num_steps):
        # Compute temporal features
        temporal_feats = _compute_temporal_features(current_time)

        # Calculate lags and rolling means based on synthetic history
        lag_feats = _calculate_lags_and_rolling(
            synthetic_history_df, current_time - timedelta(minutes=15)
        )

        # Build feature vector
        features_dict = {
            "PULocationID": zone_id,
            **temporal_feats,
            "is_holiday": is_holiday,
            "cbd_pricing_active": cbd_pricing_active,
            "is_airport_zone": is_airport,
            "borough_id": last_borough_id,
            "service_zone_id": last_service_zone_id,
            "zone_slot_baseline": zone_slot_baseline,
            **lag_feats,
        }

        # Create feature dataframe in correct order
        X_pred = pd.DataFrame([features_dict])[FEATURES]

        # Make prediction
        pred_value = _lgbm_model.predict(X_pred)[0]
        pred_value = max(0.0, pred_value)  # Clip to non-negative

        predictions.append(
            {
                "time_bucket": current_time.isoformat(),
                "predicted_trips": round(pred_value, 2),
                "hour": temporal_feats["hour"],
                "minute": temporal_feats["minute"],
            }
        )

        # Add predicted value to synthetic history for next lag calculation
        synthetic_history_df = pd.concat(
            [
                synthetic_history_df,
                pd.DataFrame(
                    {
                        "time_bucket": [current_time],
                        "trip_count": [pred_value],
                        "borough_id": [last_borough_id],
                        "service_zone_id": [last_service_zone_id],
                        "is_airport_zone": [is_airport],
                        "zone_slot_baseline": [zone_slot_baseline],
                    }
                ),
            ],
            ignore_index=True,
        )

        current_time += timedelta(minutes=15)

    return predictions


# ── Public API ──────────────────────────────────────────────────────────────


def _is_forecast_date(date: str) -> bool:
    """Check if the given date is in the forecast period (after Feb 14, 2026)."""
    if not date:
        return False
    return pd.Timestamp(date) > REFERENCE_DATE


def get_heatmap(
    hour: int, dow: int, holiday: str = "regular", date: str = None
) -> dict:
    """
    Get demand heatmap for a given date/hour.
    If date ≤ Feb 14, 2026: returns actual historical aggregates (data_type = "actual").
    If date > Feb 14, 2026: returns historical aggregates marked as forecast (data_type = "predicted").
    """
    is_forecast = _is_forecast_date(date)

    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday

    # Always use historical aggregates (which represent typical demand patterns)
    if holiday != "regular":
        sub = _profile[
            (_profile["hour"] == hour)
            & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ]
    else:
        sub = _profile[
            (_profile["hour"] == hour)
            & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ]

    demand = {
        str(int(r["PULocationID"])): round(float(r["avg"]), 2)
        for _, r in sub.iterrows()
    }
    max_val = max(demand.values(), default=1.0)

    return {
        "demand": demand,
        "max": round(max_val, 2),
        "hour": hour,
        "dow": dow,
        "holiday": holiday,
        "date": date,
        "is_forecast": is_forecast,
        "data_type": "predicted" if is_forecast else "actual",
        "reference_date": REFERENCE_DATE.isoformat().split("T")[0],
    }


def get_kpis(hour: int, dow: int, holiday: str = "regular", date: str = None) -> dict:
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    if holiday != "regular":
        sub = _profile[
            (_profile["hour"] == hour)
            & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ].copy()
    else:
        sub = _profile[
            (_profile["hour"] == hour)
            & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ].copy()

    if sub.empty:
        return {}

    total = float(sub["avg"].sum())
    active = int((sub["avg"] > 0).sum())

    top_row = sub.loc[sub["avg"].idxmax()]
    top_id = int(top_row["PULocationID"])
    top_info = _zone_map.get(top_id, {})

    # 24-hour fleet trend
    if holiday != "regular":
        trend_by_hour = (
            _profile[
                (_profile["dayofweek"] == -1) & (_profile["holiday_name"] == holiday)
            ]
            .groupby("hour")["avg"]
            .sum()
            .sort_index()
        )
    else:
        trend_by_hour = (
            _profile[
                (_profile["dayofweek"] == dow) & (_profile["holiday_name"] == "regular")
            ]
            .groupby("hour")["avg"]
            .sum()
            .sort_index()
        )
    hour_trend = [round(float(trend_by_hour.get(h, 0))) for h in range(24)]

    # Borough breakdown
    sub2 = sub.merge(
        _zones_df[["zone_id", "borough"]],
        left_on="PULocationID",
        right_on="zone_id",
        how="left",
    )
    borough = {}
    if total > 0:
        for b, grp in sub2.groupby("borough"):
            borough[str(b)] = round(float(grp["avg"].sum() / total * 100), 1)

    # vs previous hour
    if hour > 0:
        prev = float(
            _profile[(_profile["hour"] == hour - 1) & (_profile["dayofweek"] == dow)][
                "avg"
            ].sum()
        )
        vs_prev = round((total - prev) / prev * 100, 1) if prev > 0 else 0.0
    else:
        vs_prev = 0.0

    return {
        "total_trips": round(total),
        "active_zones": active,
        "peak_zone_id": top_id,
        "peak_zone_name": top_info.get("name", f"Zone {top_id}"),
        "peak_zone_trips": round(float(top_row["avg"])),
        "hour_trend": hour_trend,
        "borough_breakdown": borough,
        "vs_prev_hour_pct": vs_prev,
    }


def get_ranking(
    hour: int, dow: int, n: int = 15, holiday: str = "regular", date: str = None
) -> list:
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    if holiday != "regular":
        sub = _profile[
            (_profile["hour"] == hour)
            & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ].copy()
        prev_hour_filter = _profile[
            (_profile["hour"] == max(0, hour - 1))
            & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ]
    else:
        sub = _profile[
            (_profile["hour"] == hour)
            & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ].copy()
        prev_hour_filter = _profile[
            (_profile["hour"] == max(0, hour - 1))
            & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ]

    top = sub.nlargest(n, "avg")
    max_val = float(top["avg"].max()) if len(top) else 1.0

    prev_hour = prev_hour_filter.set_index("PULocationID")["avg"]

    result = []
    for i, (_, row) in enumerate(top.iterrows()):
        zid = int(row["PULocationID"])
        info = _zone_map.get(zid, {})
        prev = float(prev_hour.get(zid, row["avg"]))
        avg = float(row["avg"])
        trend = "up" if avg > prev + 0.1 else ("down" if avg < prev - 0.1 else "flat")
        result.append(
            {
                "rank": i + 1,
                "zone_id": zid,
                "name": info.get("name", f"Zone {zid}"),
                "borough": info.get("borough", ""),
                "trips": round(avg, 1),
                "pct_of_max": round(avg / max_val * 100),
                "trend": trend,
                "is_airport": zid in AIRPORT_ZONES,
            }
        )
    return result


def get_zone_trend(
    zone_id: int, dow: int, holiday: str = "regular", date: str = None
) -> list:
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    if holiday != "regular":
        sub = _profile[
            (_profile["PULocationID"] == zone_id)
            & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ].sort_values("hour")
    else:
        sub = _profile[
            (_profile["PULocationID"] == zone_id)
            & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ].sort_values("hour")
    return [
        {"hour": int(r["hour"]), "trips": round(float(r["avg"]), 1)}
        for _, r in sub.iterrows()
    ]


def get_recommendations(
    zone_id: int,
    hour: int,
    dow: int,
    n: int = 3,
    holiday: str = "regular",
    date: str = None,
) -> list:
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday
    if holiday != "regular":
        sub = _profile[
            (_profile["hour"] == hour)
            & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ].copy()
    else:
        sub = _profile[
            (_profile["hour"] == hour)
            & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ].copy()

    max_val = float(sub["avg"].max()) if len(sub) else 1.0

    # Check current zone demand
    current_zone_data = sub[sub["PULocationID"] == zone_id]
    current_zone_demand = (
        float(current_zone_data["avg"].values[0]) if len(current_zone_data) > 0 else 0
    )
    current_zone_is_hot = current_zone_demand / max_val > 0.7  # > 70% of max demand

    # Get candidates (exclude airports, optionally exclude current zone if low demand)
    candidates = sub[~sub["PULocationID"].isin(AIRPORT_ZONES)]
    if not current_zone_is_hot:
        candidates = candidates[candidates["PULocationID"] != zone_id]

    result = []

    # If current zone is hot, show it as first recommendation (stay put)
    if current_zone_is_hot and len(current_zone_data) > 0:
        avg_fare = _get_zone_hour_fare(zone_id, hour)
        min_trips = 1
        max_trips = min(4, int(current_zone_demand))

        info = _zone_map.get(zone_id, {})
        result.append(
            {
                "rank": 1,
                "zone_id": zone_id,
                "name": f"Stay: {info.get('name', f'Zone {zone_id}')}",
                "borough": info.get("borough", ""),
                "trips": round(current_zone_demand, 1),
                "demand_score": 100,
                "drive_minutes": 0,  # already here
                "est_yield_min": int(min_trips * avg_fare),
                "est_yield_max": int(max_trips * avg_fare),
                "efficiency_score": 999,  # highest priority (no travel)
            }
        )

    # Score remaining candidates by efficiency (earnings per minute including drive time)
    scored = []
    for _, row in candidates.iterrows():
        zid = int(row["PULocationID"])
        avg = float(row["avg"])

        avg_fare = _get_zone_hour_fare(zid, hour)
        drive_mins = _get_drive_time(zone_id, zid)

        # Assume ~15 min per pickup/service + 5 min wait = 20 min service time
        service_time_mins = 20
        total_time_mins = drive_mins + service_time_mins

        # Revenue per minute (max earnings / total time)
        max_earn = min(4, int(avg)) * avg_fare
        efficiency = max_earn / total_time_mins if total_time_mins > 0 else 0

        min_trips = 1
        max_trips = min(4, int(avg))

        info = _zone_map.get(zid, {})
        scored.append(
            {
                "rank": 0,  # will be set below
                "zone_id": zid,
                "name": info.get("name", f"Zone {zid}"),
                "borough": info.get("borough", ""),
                "trips": round(avg, 1),
                "demand_score": round(avg / max_val * 100),
                "drive_minutes": drive_mins,
                "est_yield_min": int(min_trips * avg_fare),
                "est_yield_max": int(max_trips * avg_fare),
                "efficiency_score": efficiency,
            }
        )

    # Sort by efficiency (highest first), then by demand
    scored.sort(key=lambda x: (-x["efficiency_score"], -x["demand_score"]))

    # Add top n-1 (or n if current zone not included) to result
    remaining_slots = n - len(result)
    for i, rec in enumerate(scored[:remaining_slots]):
        rec["rank"] = len(result) + 1
        result.append(rec)

    return result


def get_current_zone(
    zone_id: int, hour: int, dow: int, holiday: str = "regular", date: str = None
) -> dict:
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday

    info = _zone_map.get(zone_id, {})

    if holiday != "regular":
        sub = _profile[
            (_profile["PULocationID"] == zone_id)
            & (_profile["hour"] == hour)
            & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ]
        hour_max_filter = _profile[
            (_profile["hour"] == hour)
            & (_profile["dayofweek"] == -1)
            & (_profile["holiday_name"] == holiday)
        ]
    else:
        sub = _profile[
            (_profile["PULocationID"] == zone_id)
            & (_profile["hour"] == hour)
            & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ]
        hour_max_filter = _profile[
            (_profile["hour"] == hour)
            & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == "regular")
        ]

    avg = round(float(sub["avg"].values[0]), 1) if len(sub) else 0.0
    hour_max = float(hour_max_filter["avg"].max())
    pct = round(avg / hour_max * 100) if hour_max > 0 else 0
    level = "High" if pct >= 65 else ("Medium" if pct >= 35 else "Low")
    return {
        "zone_id": zone_id,
        "name": info.get("name", f"Zone {zone_id}"),
        "borough": info.get("borough", ""),
        "trips": avg,
        "demand_pct": pct,
        "demand_level": level,
        "holiday": holiday,
    }


def get_zone_metadata() -> list:
    return _zones_df.to_dict("records")


def _get_demand_and_forecast(
    zone_id: int, hour: int, dow: int, date: str = None, holiday: str = "regular"
):
    """
    Returns both actual demand (if date ≤ Feb 14) and forecast (if date > Feb 14).
    Also computes trend direction from LightGBM predictions.

    Returns dict:
    {
        'demand_current': float,      # actual if date <= REF, else None
        'demand_next_hour': float,    # actual if date <= REF, else LightGBM forecast
        'is_forecast': bool,          # True if using LightGBM, False if actual data
        'change_pct': int,            # % change current → next
        'trend_direction': str,       # 'rising' | 'stable' | 'falling' (4h outlook)
    }
    """
    REFERENCE_DATE = pd.Timestamp("2026-02-14")
    query_date = pd.Timestamp(date) if date else REFERENCE_DATE
    is_future = query_date > REFERENCE_DATE

    # Current demand: lookup from _profile
    current_row = _profile[
        (_profile["PULocationID"] == zone_id)
        & (_profile["hour"] == hour)
        & (_profile["dayofweek"] == dow)
        & (_profile["holiday_name"] == holiday)
    ]
    demand_current = float(current_row["avg"].iloc[0]) if len(current_row) > 0 else 0.0

    # Next hour demand
    if not is_future:
        # Use actual data for current/past dates
        next_hour = (hour + 1) % 24
        next_row = _profile[
            (_profile["PULocationID"] == zone_id)
            & (_profile["hour"] == next_hour)
            & (_profile["dayofweek"] == dow)
            & (_profile["holiday_name"] == holiday)
        ]
        demand_next = (
            float(next_row["avg"].iloc[0]) if len(next_row) > 0 else demand_current
        )
        is_forecast_used = False
        trend_direction = "stable"  # No forecast for historical data
    else:
        # Use LightGBM for future dates
        preds = forecast_demand(zone_id, hour, dow, num_steps=4, date=date)
        # Average next 4 steps (1 hour of 15-min intervals)
        demand_next = (
            (sum(p["predicted_trips"] for p in preds) / len(preds))
            if preds
            else demand_current
        )
        is_forecast_used = True

        # Compute trend direction (4h outlook)
        preds_4h = forecast_demand(zone_id, hour, dow, num_steps=16, date=date)
        if len(preds_4h) >= 16:
            avg_1h = sum(p["predicted_trips"] for p in preds_4h[0:4]) / 4
            avg_4h = sum(p["predicted_trips"] for p in preds_4h[12:16]) / 4
            if avg_4h > avg_1h * 1.1:
                trend_direction = "rising"
            elif avg_4h < avg_1h * 0.9:
                trend_direction = "falling"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "stable"

    change_pct = (
        round((demand_next - demand_current) / demand_current * 100)
        if demand_current > 0
        else 0
    )

    return {
        "demand_current": demand_current,
        "demand_next_hour": demand_next,
        "is_forecast": is_forecast_used,
        "change_pct": change_pct,
        "trend_direction": trend_direction,
    }


def get_operator_zones(
    hour: int, dow: int, date: str = None, holiday: str = "regular"
) -> list:
    """
    Operator dashboard: zones with demand, forecast, unmet demand, and action guidance.
    Uses LightGBM for future dates, actual data for historical.
    """
    if _profile is None:
        return []

    REFERENCE_DATE = pd.Timestamp("2026-02-14")
    query_date = pd.Timestamp(date) if date else REFERENCE_DATE
    is_future = query_date > REFERENCE_DATE

    # Auto-detect holiday if date provided
    if date:
        _, _, _, detected_holiday = _get_week_context(date)
        if detected_holiday != "regular":
            holiday = detected_holiday

    zones_data = []

    for zone_id in _profile["PULocationID"].unique():
        # Get current and next-hour demand (actual or forecast)
        demand_data = _get_demand_and_forecast(zone_id, hour, dow, date, holiday)
        demand_now = demand_data["demand_current"]
        demand_next = demand_data["demand_next_hour"]
        change_pct = demand_data["change_pct"]
        is_using_forecast = demand_data["is_forecast"]
        trend_direction = demand_data["trend_direction"]

        # Get zone metadata
        zone_info = _zone_map.get(zone_id, {})

        # === UNMET DEMAND (DATA-DRIVEN) ===
        # Use 95th percentile of historical demand as the stress threshold
        # Demand above that indicates the zone is beyond "normal peak"
        p95_baseline = _zone_unmet_baselines.get(zone_id, 0)

        if p95_baseline > 0 and demand_now > p95_baseline:
            # Zone exceeds its 95th percentile (normal peak)
            # Estimate overflow as potentially unmet demand
            overflow = demand_now - p95_baseline
            # Assume 10-30% of overflow goes unfulfilled (grows with severity)
            overflow_ratio = overflow / p95_baseline  # How much above p95
            unmet_pct = min(0.30, 0.10 + overflow_ratio * 0.20)  # 10-30% unmet
            unmet_est = max(0, round(overflow * unmet_pct))
        else:
            unmet_est = 0

        # Volatility signal
        volatility_ratio = _zone_volatility.get((zone_id, hour), 0.0)
        if volatility_ratio > 0.3:
            volatility_label = "high"
        elif volatility_ratio > 0.15:
            volatility_label = "medium"
        else:
            volatility_label = "low"

        # Supply status
        # Tight: any unmet demand (exceeds 90th percentile baseline)
        # Balanced: high demand (> 30 trips/hr) but no unmet
        # Light: normal demand
        if unmet_est > 0:
            supply_status = "tight"
        elif demand_now > 30:
            supply_status = "balanced"
        else:
            supply_status = "light"

        # Revenue potential
        base_revenue = demand_now * 13.5
        drivers_needed = max(1, round(demand_now / 25))

        # Action recommendation (rule-based)
        action = None
        if supply_status == "tight" and volatility_label == "high":
            action = "High demand + volatile. Raise surge cautiously."
        elif supply_status == "tight" and volatility_label == "low":
            action = "High demand + stable. Can raise surge aggressively."
        elif trend_direction == "rising" and supply_status == "balanced":
            action = "Demand rising. Prepare extra drivers."
        elif trend_direction == "falling" and supply_status == "tight":
            action = "Demand falling. Monitor before surge."

        zones_data.append(
            {
                "zone_id": int(zone_id),
                "name": zone_info.get("name", f"Zone {zone_id}"),
                "borough": zone_info.get("borough", ""),
                "demand_now": round(float(demand_now), 1),
                "demand_next_hour": round(float(demand_next), 1),
                "change_pct": int(change_pct),
                "unmet_demand": int(unmet_est),
                "drivers_needed": int(drivers_needed),
                "revenue_potential": int(round(base_revenue)),
                "supply_status": supply_status,
                "is_forecast": bool(is_using_forecast),
                "forecast_source": "lgbm" if is_using_forecast else "actual",
                "volatility_label": volatility_label,
                "volatility_score": round(float(volatility_ratio), 2),
                "trend_direction": trend_direction,
                "action": action,
            }
        )

    # Sort by opportunity: unmet_demand descending, then demand_now descending
    zones_data.sort(key=lambda z: (z["unmet_demand"], z["demand_now"]), reverse=True)
    return zones_data


def get_forecast_heatmap(hours_ahead: int = 0) -> dict:
    """
    Get forecasted demand heatmap for all zones at a specific time ahead.
    Uses synthetic current demand as baseline with intelligent variations.

    Args:
        hours_ahead: Hours into the future (0-4)

    Returns:
        Heatmap dict with forecasted demand for all zones
    """
    if _lgbm_model is None or _full_demand is None:
        return {
            "demand": {},
            "max": 0.0,
            "hours_ahead": hours_ahead,
            "is_forecast": True,
        }

    # Get current time reference
    now = pd.Timestamp.now()
    current_hour = now.hour
    current_dow = now.dayofweek

    # Calculate number of 15-min steps
    steps_ahead = max(1, (hours_ahead * 60) // 15)

    # Get all unique zones
    all_zones = _full_demand["PULocationID"].unique()

    # Forecast for each zone and extract the prediction at steps_ahead
    demand = {}
    max_val = 0.0

    for zone_id in all_zones:
        forecast = forecast_demand(
            int(zone_id), current_hour, current_dow, num_steps=steps_ahead
        )
        if forecast and len(forecast) >= steps_ahead:
            predicted = forecast[steps_ahead - 1]["predicted_trips"]
            demand[str(int(zone_id))] = round(predicted, 2)
            max_val = max(max_val, predicted)

    return {
        "demand": demand,
        "max": round(max_val, 2),
        "hours_ahead": hours_ahead,
        "is_forecast": True,
    }
