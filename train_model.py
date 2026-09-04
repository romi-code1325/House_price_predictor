"""
train_model.py
Generates a realistic synthetic real-estate dataset (no internet needed —
important for a hackathon demo where you can't risk a flaky download),
trains a Random Forest regressor, and saves the model for app.py to use.

Run this FIRST, before running app.py:
    python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

MODEL_PATH = "model.joblib"
DATA_PATH = "housing_data.csv"
N_SAMPLES = 3000

# Reference "city center" coordinates used to place synthetic listings on
# a map. Defaults to Nagpur, India — change to your own city if you like.
CITY_CENTER_LAT = 21.1458
CITY_CENTER_LON = 79.0882

# Name pools for generating fake (but readable) seller contact details.
_FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Ishaan", "Rohan", "Kabir", "Arjun", "Sai",
    "Ananya", "Diya", "Priya", "Neha", "Kavya", "Meera", "Riya", "Sneha",
    "Rahul", "Amit", "Vikram", "Sanjay", "Pooja", "Anjali", "Ritu", "Suman",
]
_LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Gupta", "Reddy", "Iyer", "Nair", "Rao",
    "Joshi", "Mehta", "Kulkarni", "Deshmukh", "Agarwal", "Chauhan", "Bose",
]
_STREET_NAMES = [
    "MG Road", "Park Avenue", "Lake View Lane", "Church Street", "Ring Road",
    "Station Road", "Green Valley Road", "Civil Lines", "Hill Top Road",
    "Sunrise Boulevard", "College Road", "Market Street",
]
_AREA_NAMES = [
    "Dharampeth", "Sadar", "Civil Lines", "Sitabuldi", "Ramdaspeth",
    "Manish Nagar", "Wardha Road", "Trimurti Nagar", "Hingna Road",
    "Pratap Nagar", "Gandhibagh", "Laxmi Nagar",
]


def _generate_contact_details(n, rng):
    """Generates a plausible fake name, phone number, and address per row."""
    first = rng.choice(_FIRST_NAMES, n)
    last = rng.choice(_LAST_NAMES, n)
    names = [f"{f} {l}" for f, l in zip(first, last)]

    # Indian-style 10-digit mobile numbers, starting with 6-9
    phone_prefix = rng.integers(6, 10, n)
    phone_rest = rng.integers(0, 10, size=(n, 9))
    phones = [
        f"+91 {p}{''.join(str(d) for d in phone_rest[i])[:5]} "
        f"{''.join(str(d) for d in phone_rest[i])[5:]}"
        for i, p in enumerate(phone_prefix)
    ]

    house_no = rng.integers(1, 999, n)
    streets = rng.choice(_STREET_NAMES, n)
    areas = rng.choice(_AREA_NAMES, n)
    addresses = [
        f"{h}, {s}, {a}" for h, s, a in zip(house_no, streets, areas)
    ]

    return names, phones, addresses


def generate_data(n=N_SAMPLES, seed=42):
    """
    Create a synthetic but realistic housing dataset.
    Price is built from a formula with real-world-plausible relationships
    (bigger, newer, more central homes cost more) plus noise, so the model
    has genuine signal to learn instead of pure randomness.
    """
    rng = np.random.default_rng(seed)

    sqft = rng.normal(1800, 650, n).clip(400, 6000)
    bedrooms = rng.integers(1, 6, n)
    bathrooms = rng.integers(1, 4, n) + rng.choice([0, 0.5], n)
    house_age = rng.integers(0, 80, n)
    distance_to_city_km = rng.exponential(8, n).clip(0.5, 60)
    school_rating = rng.integers(1, 11, n)  # 1-10
    crime_index = rng.uniform(0, 100, n)     # lower is safer

    # Base price formula (in $1000s) with realistic weights
    price = (
        50_000
        + sqft * 120
        + bedrooms * 8_000
        + bathrooms * 6_000
        - house_age * 500
        - distance_to_city_km * 1_800
        + school_rating * 4_500
        - crime_index * 350
    )
    # Add noise (measurement error / market randomness)
    price += rng.normal(0, 25_000, n)
    price = price.clip(40_000, None)  # no negative/absurdly low prices

    # Place each listing on a map: pick a random bearing around the city
    # center and walk out by DistanceToCityKm. Rough conversion (good
    # enough for a demo, not survey-grade): 1 degree lat ~= 111 km.
    angle = rng.uniform(0, 2 * np.pi, n)
    km_per_deg_lat = 111.0
    km_per_deg_lon = 111.0 * np.cos(np.radians(CITY_CENTER_LAT))
    lat = CITY_CENTER_LAT + (distance_to_city_km * np.sin(angle)) / km_per_deg_lat
    lon = CITY_CENTER_LON + (distance_to_city_km * np.cos(angle)) / km_per_deg_lon

    names, phones, addresses = _generate_contact_details(n, rng)

    df = pd.DataFrame({
        "SqFt": sqft.round(0),
        "Bedrooms": bedrooms,
        "BHK": bedrooms,  # same as bedrooms; kept as a separate labeled column for display
        "Bathrooms": bathrooms,
        "HouseAgeYears": house_age,
        "DistanceToCityKm": distance_to_city_km.round(1),
        "SchoolRating": school_rating,
        "CrimeIndex": crime_index.round(1),
        "Latitude": lat.round(5),
        "Longitude": lon.round(5),
        "SellerName": names,
        "ContactNumber": phones,
        "Address": addresses,
        "PriceUSD": price.round(0),
    })
    return df


def train():
    print("Generating synthetic housing dataset (offline, no download)...")
    df = generate_data()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved dataset to {DATA_PATH} ({len(df)} rows)")

    # These columns are for display only (map, contact card) — not model
    # inputs. BHK duplicates Bedrooms; SellerName/ContactNumber/Address
    # are non-numeric identifiers.
    non_feature_cols = {
        "PriceUSD", "Latitude", "Longitude", "BHK",
        "SellerName", "ContactNumber", "Address",
    }
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    X = df[feature_cols]
    y = df["PriceUSD"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training baseline Linear Regression...")
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_preds = lin_model.predict(X_test)
    print(f"  Linear R2: {r2_score(y_test, lin_preds):.3f} | "
          f"MAE: ${mean_absolute_error(y_test, lin_preds):,.0f}")

    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_preds)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    print(f"  Random Forest R2: {rf_r2:.3f} | MAE: ${rf_mae:,.0f}")

    bundle = {
        "model": rf_model,
        "feature_cols": feature_cols,
        "metrics": {"r2": rf_r2, "mae": rf_mae},
        "feature_ranges": {
            col: (float(X[col].min()), float(X[col].max()), float(X[col].median()))
            for col in feature_cols
        },
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")
    print("Done. Now run: streamlit run app.py")


if __name__ == "__main__":
    train()
