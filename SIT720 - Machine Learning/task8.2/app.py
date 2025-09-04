# app.py
# ------------------------------------------------------------
# Step 5: Simple web demo
#   - Loads price_model.pkl (pipeline)
#   - Lets users input features
#   - Returns price prediction
# Run: streamlit run app.py
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Housing Price Predictor", page_icon="🏡", layout="centered")
st.title("🏡 Housing Price Prediction")

# Load trained pipeline & metadata
art = joblib.load("price_model.pkl")
pipe = art["pipeline"]
feature_list = art["feature_list"]
categorical_features = art["categorical_features"]
numeric_features = art["numeric_features"]

# To populate dropdowns nicely, we’ll load the original dataset
raw = pd.read_csv("dataset.csv")
if "address" in raw.columns: raw = raw.drop(columns=["address"])
if "state"   in raw.columns: raw = raw.drop(columns=["state"])

# Prepare options for categorical fields
for col in ["suburb","postcode","type"]:
    if col in raw.columns:
        raw[col] = raw[col].fillna(raw[col].mode()[0])

suburb_opts  = sorted(raw["suburb"].dropna().astype(str).unique())  if "suburb"  in raw.columns else []
postcode_opts= sorted(raw["postcode"].dropna().astype(str).unique())if "postcode" in raw.columns else []
type_opts    = sorted(raw["type"].dropna().astype(str).unique())    if "type"    in raw.columns else []

with st.form("input-form"):
    st.subheader("Enter Property Details")
    c1, c2 = st.columns(2)
    with c1:
        suburb   = st.selectbox("Suburb", suburb_opts) if suburb_opts else st.text_input("Suburb")
        postcode = st.selectbox("Postcode", postcode_opts) if postcode_opts else st.text_input("Postcode")
        ptype    = st.selectbox("Property Type", type_opts) if type_opts else st.text_input("Type (House/Townhouse/Unit)")
    with c2:
        bedroom  = st.number_input("Bedrooms", min_value=0, max_value=10, value=4, step=1)
        bathroom = st.number_input("Bathrooms", min_value=0, max_value=10, value=2, step=1)
        carspace = st.number_input("Car Spaces", min_value=0, max_value=6, value=2, step=1)

    size = st.number_input("Size (sqm)", min_value=0, max_value=2000, value=650, step=10)

    # Date-related inputs (we’ll derive engineered features from these)
    st.markdown("**Sale Date**")
    sale_date = st.date_input("Select a sale date", None)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    # Build a single-row DataFrame with ALL features used during training
    row = {f: np.nan for f in feature_list}

    # Fill directly observed fields
    row["suburb"]   = str(suburb) if "suburb" in feature_list else None
    row["postcode"] = str(postcode) if "postcode" in feature_list else None
    row["type"]     = str(ptype) if "type" in feature_list else None

    row["bedroom"]  = float(bedroom) if "bedroom" in feature_list else None
    row["bathroom"] = float(bathroom) if "bathroom" in feature_list else None
    row["carspace"] = float(carspace) if "carspace" in feature_list else None
    row["size"]     = float(size) if "size" in feature_list else None

    # Engineered features
    if sale_date is not None:
        row["sale_year"]  = sale_date.year if "sale_year" in feature_list else None
        row["sale_month"] = sale_date.month if "sale_month" in feature_list else None
        # Monday=0 .. Sunday=6
        row["sale_dow"]   = sale_date.weekday() if "sale_dow" in feature_list else None
        row["is_weekend"] = 1 if sale_date.weekday() >= 5 else 0 if "is_weekend" in feature_list else None
    else:
        # Fallback reasonable defaults
        if "sale_year" in feature_list:  row["sale_year"]  = 2025
        if "sale_month" in feature_list: row["sale_month"] = 8
        if "sale_dow" in feature_list:   row["sale_dow"]   = 5
        if "is_weekend" in feature_list: row["is_weekend"] = 1

    # rooms_total / size_per_room / carspace_per_rm
    rooms_total = max(1.0, float(bedroom) + float(bathroom))
    if "rooms_total" in feature_list:   row["rooms_total"] = rooms_total
    if "size_per_room" in feature_list: row["size_per_room"] = float(size) / rooms_total if rooms_total > 0 else 0.0
    if "carspace_per_rm" in feature_list: row["carspace_per_rm"] = float(carspace) / rooms_total if rooms_total > 0 else 0.0

    # frequency encodings (approximate using current raw distribution)
    if "suburb_freq" in feature_list and "suburb" in raw.columns:
        row["suburb_freq"] = int((raw["suburb"] == suburb).sum())
    if "postcode_freq" in feature_list and "postcode" in raw.columns:
        row["postcode_freq"] = int((raw["postcode"] == postcode).sum())

    X_input = pd.DataFrame([row], columns=feature_list)

    # Predict
    try:
        price_pred = pipe.predict(X_input)[0]
        st.success(f"Estimated Price: ${price_pred:,.0f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.caption("Note: This is a demo model trained on the provided dataset; real-world predictions may differ.")