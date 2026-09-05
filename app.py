"""
app.py
Streamlit demo app for the Real Estate Price Predictor.
Includes a lightweight login/signup gate and a map tab showing nearby
synthetic listings, so you can "click around" a location and estimate
price near it.

Run:
    streamlit run app.py

Requires model.joblib and housing_data.csv, which are created by
running train_model.py first.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

import auth
import train_model

MODEL_PATH = "model.joblib"
DATA_PATH = "housing_data.csv"

# Must match CITY_CENTER_LAT / CITY_CENTER_LON in train_model.py
CITY_CENTER_LAT = 21.1458
CITY_CENTER_LON = 79.0882

st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# On a fresh deploy (e.g. Streamlit Community Cloud) housing_data.csv and
# model.joblib won't exist yet, since only app.py gets run automatically —
# train_model.py normally has to be run manually first. This trains it
# automatically on first load instead, so hosting "just works".
if not (os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH)):
    with st.spinner("First-time setup: training the model, this takes a few seconds..."):
        train_model.train()


# ---------------------------------------------------------------------
# Auth gate — split-screen branded design (left: hero panel, right: form)
# ---------------------------------------------------------------------
def _inject_login_css():
    st.markdown("""
        <style>
        /* Hide default Streamlit chrome on the login screen for a cleaner look */
        [data-testid="stSidebar"] { display: none; }
        .stApp { background: #f3f4f6; }
        .block-container { padding-top: 3rem; padding-bottom: 3rem; max-width: 1100px; }

        .hero-panel {
            background: linear-gradient(135deg, #1e3a5f 0%, #2c5282 45%, #d97706 130%);
            border-radius: 20px;
            padding: 3rem 2.4rem;
            height: 100%;
            min-height: 500px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            color: white;
            box-shadow: 0 20px 40px -12px rgba(30, 58, 95, 0.45);
            position: relative;
            overflow: hidden;
        }
        .hero-panel::before {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.08);
            top: -80px;
            right: -80px;
        }
        .hero-panel::after {
            content: "";
            position: absolute;
            width: 180px;
            height: 180px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.06);
            bottom: -60px;
            left: -60px;
        }
        .hero-icon { font-size: 2.6rem; margin-bottom: 1rem; }
        .hero-eyebrow {
            letter-spacing: 3px;
            font-size: 0.8rem;
            font-weight: 700;
            opacity: 0.9;
            margin-bottom: 0.8rem;
            position: relative;
        }
        .hero-title {
            font-size: 2.7rem;
            font-weight: 800;
            line-height: 1.15;
            margin-bottom: 1.1rem;
            position: relative;
        }
        .hero-subtitle {
            font-size: 1rem;
            opacity: 0.92;
            line-height: 1.6;
            margin-bottom: 1.8rem;
            position: relative;
        }
        .hero-tick {
            font-size: 0.94rem;
            opacity: 0.95;
            margin-bottom: 0.5rem;
            position: relative;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        .hero-tick-badge {
            background: rgba(255, 255, 255, 0.18);
            border-radius: 50%;
            width: 20px;
            height: 20px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 0.7rem;
            flex-shrink: 0;
        }

        /* Style the right-hand column itself as a white elevated card —
           more reliable than wrapping Streamlit widgets in a custom div,
           since Streamlit renders each element as its own sibling block. */
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-of-type(2) {
            background: white;
            border-radius: 20px;
            padding: 2.6rem 2.2rem;
            box-shadow: 0 20px 40px -16px rgba(0, 0, 0, 0.12);
        }
        .form-eyebrow {
            letter-spacing: 2px;
            font-size: 0.78rem;
            font-weight: 700;
            color: #d97706;
            margin-bottom: 0.4rem;
        }
        .form-title { font-size: 2.1rem; font-weight: 800; margin-bottom: 0.35rem; color: #111827; }
        .form-caption { color: #6b7280; margin-bottom: 1.8rem; }

        div[data-testid="stFormSubmitButton"] button {
            background: linear-gradient(90deg, #d97706, #ea580c);
            border: none;
            font-weight: 700;
            padding: 0.6rem 0;
            box-shadow: 0 8px 16px -6px rgba(217, 119, 6, 0.5);
            transition: transform 0.15s ease;
        }
        div[data-testid="stFormSubmitButton"] button:hover {
            transform: translateY(-1px);
        }
        div[data-testid="stTextInput"] input {
            border-radius: 8px;
            border: 1px solid #e5e7eb;
        }
        .stTabs [data-baseweb="tab"] { font-weight: 600; }
        </style>
    """, unsafe_allow_html=True)


def login_signup_screen():
    _inject_login_css()

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("""
            <div class="hero-panel">
                <div class="hero-icon">🏠</div>
                <div class="hero-eyebrow">REAL ESTATE PRICE PREDICTOR</div>
                <div class="hero-title">Know a Home's<br>True Value<br>Before You Decide</div>
                <div class="hero-subtitle">
                    Instant, data-driven price estimates powered by machine
                    learning — trained on square footage, location, school
                    ratings, and more.
                </div>
                <div class="hero-tick"><span class="hero-tick-badge">✓</span> Instant price predictions</div>
                <div class="hero-tick"><span class="hero-tick-badge">✓</span> Interactive location map</div>
                <div class="hero-tick"><span class="hero-tick-badge">✓</span> Your prediction history saved</div>
            </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="form-eyebrow">WELCOME BACK</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-title">Sign in</div>', unsafe_allow_html=True)
        st.markdown('<div class="form-caption">Continue to your price predictor account.</div>', unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["Log in", "Sign up"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", key="login_username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In", type="primary", use_container_width=True)
                if submitted:
                    if auth.verify_user(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username.strip()
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")

        with tab_signup:
            with st.form("signup_form"):
                new_username = st.text_input("Choose a username", key="signup_username", placeholder="e.g. john_doe")
                new_password = st.text_input("Choose a password", type="password", key="signup_password", placeholder="At least 4 characters")
                confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm", placeholder="Re-enter password")
                submitted = st.form_submit_button("Create Account", type="primary", use_container_width=True)
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match.")
                    else:
                        ok, message = auth.create_user(new_username, new_password)
                        if ok:
                            st.success(message + " Switch to the Log in tab.")
                        else:
                            st.error(message)


if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    login_signup_screen()
    st.stop()


# ---------------------------------------------------------------------
# Data / model loading
# ---------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


bundle = load_model()
df = load_data()

# ---------------------------------------------------------------------
# Top bar: title + logout
# ---------------------------------------------------------------------
top_left, top_right = st.columns([4, 1])
with top_left:
    st.title("🏠 Real Estate Price Predictor")
    st.caption(f"Logged in as **{st.session_state.get('username', 'guest')}**")
with top_right:
    if st.button("Log out"):
        st.session_state.authenticated = False
        st.session_state.pop("username", None)
        st.rerun()

if bundle is None:
    st.error("No trained model found. Run `python train_model.py` first, then restart this app.")
    st.stop()

model = bundle["model"]
feature_cols = bundle["feature_cols"]
ranges = bundle["feature_ranges"]
metrics = bundle["metrics"]

tab_predict, tab_map, tab_history, tab_data = st.tabs(
    ["Predict a price", "Find prices near a location", "My history", "Dataset exploration"]
)

# ---------------------------------------------------------------------
# Tab 1: manual prediction form
# ---------------------------------------------------------------------
with tab_predict:
    left, right = st.columns([1, 1.2])

    with left:
        st.subheader("Property details")

        def slider_for(col, label, step=1.0, fmt="%.0f"):
            lo, hi, med = ranges[col]
            return st.slider(label, min_value=float(lo), max_value=float(hi),
                              value=float(med), step=step, format=fmt)

        sqft = slider_for("SqFt", "Square footage", step=10.0)
        bedrooms = st.slider("Bedrooms", 1, 6, 3)
        bathrooms = st.slider("Bathrooms", 1.0, 4.5, 2.0, step=0.5)
        house_age = slider_for("HouseAgeYears", "House age (years)", step=1.0)
        distance = slider_for("DistanceToCityKm", "Distance to city center (km)", step=0.5)
        school = st.slider("School rating (1-10)", 1, 10, 7)
        crime = slider_for("CrimeIndex", "Crime index (lower = safer)", step=1.0)

        input_row = pd.DataFrame([{
            "SqFt": sqft,
            "Bedrooms": bedrooms,
            "Bathrooms": bathrooms,
            "HouseAgeYears": house_age,
            "DistanceToCityKm": distance,
            "SchoolRating": school,
            "CrimeIndex": crime,
        }])[feature_cols]

        predict_clicked = st.button("Predict price", type="primary", use_container_width=True)

    with right:
        st.subheader("Prediction")
        if predict_clicked:
            pred = model.predict(input_row)[0]
            price_per_sqft = pred / sqft if sqft else 0
            c1, c2 = st.columns(2)
            c1.metric("Estimated price", f"${pred:,.0f}")
            c2.metric("Price per sq ft", f"${price_per_sqft:,.0f}")

            tree_preds = [t.predict(input_row)[0] for t in model.estimators_]
            lo, hi = pd.Series(tree_preds).quantile([0.1, 0.9])
            st.caption(f"Likely range: ${lo:,.0f} - ${hi:,.0f} (model uncertainty)")

            auth.save_prediction(
                st.session_state.username,
                {
                    "sqft": sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
                    "house_age": house_age, "distance_km": distance,
                    "school_rating": school, "crime_index": crime,
                },
                pred,
            )
        else:
            st.info("Set the property details on the left, then click **Predict price**.")

        st.divider()
        st.subheader("Model info")
        c1, c2 = st.columns(2)
        c1.metric("R\u00b2 (test set)", f"{metrics['r2']:.3f}")
        c2.metric("Avg error (MAE)", f"${metrics['mae']:,.0f}")

        st.subheader("What drives the prediction most?")
        importances = pd.Series(model.feature_importances_, index=feature_cols)
        importances = importances.sort_values(ascending=True)
        st.bar_chart(importances)

# ---------------------------------------------------------------------
# Tab 2: map — nearby listings + location-based price
# ---------------------------------------------------------------------
with tab_map:
    st.subheader("Find prices near a location")
    st.caption(
        "Pick a location by setting how far it is from the city center. "
        "This shows nearby synthetic listings and estimates a price for "
        "a typical home at that distance."
    )

    if df is None:
        st.warning("housing_data.csv not found - run train_model.py first.")
    else:
        map_left, map_right = st.columns([1, 1.4])

        with map_left:
            target_distance = st.slider(
                "Distance from city center (km)", 0.0, 60.0, 10.0, step=0.5
            )
            st.markdown("**Typical home used for this estimate:**")
            typ_sqft = st.slider("Square footage ", 400, 6000, 1800, step=50, key="map_sqft")
            typ_bed = st.slider("Bedrooms ", 1, 6, 3, key="map_bed")
            typ_bath = st.slider("Bathrooms ", 1.0, 4.5, 2.0, step=0.5, key="map_bath")
            typ_age = st.slider("House age (years) ", 0, 80, 15, key="map_age")
            typ_school = st.slider("School rating (1-10) ", 1, 10, 7, key="map_school")
            typ_crime = st.slider("Crime index ", 0.0, 100.0, 30.0, key="map_crime")

            map_input = pd.DataFrame([{
                "SqFt": typ_sqft,
                "Bedrooms": typ_bed,
                "Bathrooms": typ_bath,
                "HouseAgeYears": typ_age,
                "DistanceToCityKm": target_distance,
                "SchoolRating": typ_school,
                "CrimeIndex": typ_crime,
            }])[feature_cols]

            map_pred = model.predict(map_input)[0]
            st.metric(f"Estimated price at {target_distance:.1f} km from city center", f"${map_pred:,.0f}")

        with map_right:
            # Nearby listings: within +/- 5km of the chosen distance ring
            nearby = df[
                (df["DistanceToCityKm"] >= max(0, target_distance - 5))
                & (df["DistanceToCityKm"] <= target_distance + 5)
            ]
            st.caption(f"{len(nearby)} nearby listings shown on the map")
            if len(nearby) > 0:
                map_df = nearby[["Latitude", "Longitude"]].rename(
                    columns={"Latitude": "lat", "Longitude": "lon"}
                )
                st.map(map_df, size=20)
                with st.expander("View nearby listings"):
                    st.dataframe(
                        nearby[["SqFt", "Bedrooms", "Bathrooms", "DistanceToCityKm", "PriceUSD"]]
                        .sort_values("DistanceToCityKm")
                        .head(30)
                    )
            else:
                st.info("No listings found at that distance range.")

# ---------------------------------------------------------------------
# Tab 3: my prediction history
# ---------------------------------------------------------------------
with tab_history:
    st.subheader("My prediction history")

    history = auth.get_history(st.session_state.username)
    if not history:
        st.info("No predictions yet — make one in the **Predict a price** tab.")
    else:
        hist_df = pd.DataFrame(history).rename(columns={
            "created_at": "When",
            "sqft": "SqFt",
            "bedrooms": "Bedrooms",
            "bathrooms": "Bathrooms",
            "house_age": "HouseAgeYears",
            "distance_km": "DistanceToCityKm",
            "school_rating": "SchoolRating",
            "crime_index": "CrimeIndex",
            "predicted_price": "PredictedPrice",
            "price_per_sqft": "PricePerSqFt",
        })
        st.dataframe(hist_df, use_container_width=True)

        if st.button("Clear my history"):
            auth.clear_history(st.session_state.username)
            st.rerun()

# ---------------------------------------------------------------------
# Tab 4: dataset exploration — click a point to see listing contact info
# ---------------------------------------------------------------------
def _render_listing_card(row: pd.Series):
    st.markdown("#### Listing details")
    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Seller:** {row['SellerName']}")
        st.write(f"**Contact number:** {row['ContactNumber']}")
        st.write(f"**Address:** {row['Address']}")
    with c2:
        st.write(f"**Price:** ${row['PriceUSD']:,.0f}")
        st.write(f"**BHK:** {int(row['BHK'])}")
        st.write(f"**Square footage:** {row['SqFt']:,.0f} sq ft")
        st.write(f"**Distance to city center:** {row['DistanceToCityKm']} km")


with tab_data:
    st.subheader("Dataset exploration")
    st.caption("Click a point on either chart to see that listing's contact details.")

    if df is not None:
        c1, c2 = st.tabs(["Price vs. Square Footage", "Price vs. Distance to City"])

        with c1:
            fig_sqft = px.scatter(
                df, x="SqFt", y="PriceUSD",
                hover_data=["BHK", "Address"],
                labels={"SqFt": "Square footage", "PriceUSD": "Price (USD)"},
            )
            event_sqft = st.plotly_chart(
                fig_sqft, use_container_width=True,
                on_select="rerun", key="scatter_sqft",
            )
            if event_sqft and event_sqft.selection and event_sqft.selection.points:
                idx = event_sqft.selection.points[0]["point_index"]
                _render_listing_card(df.iloc[idx])
            else:
                st.info("No point selected yet.")

        with c2:
            fig_dist = px.scatter(
                df, x="DistanceToCityKm", y="PriceUSD",
                hover_data=["BHK", "Address"],
                labels={"DistanceToCityKm": "Distance to city center (km)", "PriceUSD": "Price (USD)"},
            )
            event_dist = st.plotly_chart(
                fig_dist, use_container_width=True,
                on_select="rerun", key="scatter_distance",
            )
            if event_dist and event_dist.selection and event_dist.selection.points:
                idx = event_dist.selection.points[0]["point_index"]
                _render_listing_card(df.iloc[idx])
            else:
                st.info("No point selected yet.")

        with st.expander("View raw data sample"):
            st.dataframe(df.sample(20, random_state=1))
    else:
        st.warning("housing_data.csv not found - run train_model.py first.")
