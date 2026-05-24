"""
app.py -- Interactive Auto-Insurance Claim Risk Scorer
======================================================
A tiny deployment layer on top of the trained model. Enter a policy's attributes and
get back an estimated probability of filing a claim, plus a plain-language risk band.

Run locally:        streamlit run app.py
Deploy (free):      push to GitHub -> share.streamlit.io -> point at this file.

Note: run the notebook first so `claim_model.joblib` exists. If it's missing, the app
trains a model on the fly from claim_pipeline so it still works out of the box.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st

import claim_pipeline as cp

st.set_page_config(page_title="Auto Claim Risk Scorer", page_icon="🚗", layout="centered")


@st.cache_resource
def get_model():
    """Load the saved model, or train one on the fly as a fallback."""
    if os.path.exists("claim_model.joblib"):
        import joblib
        return joblib.load("claim_model.joblib"), "loaded from claim_model.joblib"
    from sklearn.model_selection import train_test_split
    df, _ = cp.load_data()
    X, y, _ = cp.prepare(df)
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.25,
                                      random_state=cp.RANDOM_STATE, stratify=y)
    model = cp.train_models(Xtr, ytr)["XGBoost"]
    return model, "trained on the fly"


model, model_status = get_model()

st.title("🚗 Auto-Insurance Claim Risk Scorer")
st.caption(
    "Estimates the probability a policy files **at least one claim** in a year — the "
    "*frequency* component of insurance pricing. Demo on public data; not for real pricing."
)
st.markdown("---")

st.subheader("Policy & driver attributes")
c1, c2 = st.columns(2)
with c1:
    driv_age = st.slider("Driver age", 18, 90, 35)
    veh_age = st.slider("Vehicle age (years)", 0, 25, 5)
    veh_power = st.slider("Vehicle power (rating)", 4, 15, 6)
    bonus_malus = st.slider("Bonus-Malus score (50 = best, 230 = worst)", 50, 230, 60)
with c2:
    density = st.select_slider(
        "Area population density (people/km²)",
        options=[10, 50, 100, 300, 800, 2000, 5000, 12000, 27000], value=300)
    veh_gas = st.selectbox("Fuel type", ["Regular", "Diesel"])
    area = st.selectbox("Area category", list("ABCDEF"), index=2)
    region = st.selectbox("Region", [f"R{i}" for i in range(1, 23)], index=0)
veh_brand = st.selectbox("Vehicle brand", [f"B{i}" for i in range(1, 15)], index=0)

row = pd.DataFrame([{
    "VehPower": veh_power, "VehAge": veh_age, "DrivAge": driv_age,
    "BonusMalus": bonus_malus, "Density": density,
    "VehBrand": veh_brand, "VehGas": veh_gas, "Region": region, "Area": area,
}])[cp.ALL_FEATURES]

if st.button("Score this policy", type="primary"):
    p = float(model.predict_proba(row)[:, 1][0])
    st.markdown("---")
    st.metric("Estimated probability of a claim", f"{p:.1%}")
    st.progress(min(p / 0.3, 1.0))  # scale bar: 30% is already very high here
    if p < 0.06:
        st.success("**Low risk band** — below the typical portfolio claim rate.")
    elif p < 0.12:
        st.info("**Moderate risk band** — around the typical portfolio claim rate.")
    else:
        st.warning("**Elevated risk band** — above the typical portfolio claim rate.")
    st.caption(
        "Operating thresholds in a real system would be set by the *business* cost of a "
        "false positive vs. a missed risk — not fixed at arbitrary cutoffs."
    )

st.markdown("---")
st.caption(f"Model: {model_status}.  Frequency-only demo — a full pricing model also needs "
           "claim severity. Built as interview prep, not a production pricing tool.")
