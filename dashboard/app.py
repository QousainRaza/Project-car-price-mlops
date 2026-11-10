"""Step 7: the page people see.   Run locally:  streamlit run dashboard/app.py"""
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://127.0.0.1:8000")
REPORTS = Path(__file__).resolve().parents[1] / "reports"

DEFAULT_CHOICES = {
    "brands": ["Toyota", "Suzuki", "Honda"],
    "models_by_brand": {"Toyota": ["Toyota Corolla"], "Suzuki": ["Suzuki Alto"], "Honda": ["Honda Civic"]},
    "cities": ["Lahore", "Karachi", "Islamabad"],
    "fuels": ["Petrol", "Diesel", "Hybrid"],
    "transmissions": ["Manual", "Automatic"],
}


def read_json(name, default=None):
    f = REPORTS / name
    try:
        return json.loads(f.read_text()) if f.exists() else default
    except Exception:
        return default


choices = read_json("choices.json", DEFAULT_CHOICES)

st.set_page_config(page_title="Used Car Price Estimator", layout="wide")
st.title("Used Car Price Estimator")

# ---------------- sidebar: system status ----------------
st.sidebar.header("System status")
try:
    h = requests.get(f"{API}/health", timeout=3).json()
    st.sidebar.write("Model version:", h["model_version"])
    if not h["model_loaded"]:
        st.sidebar.warning("No model loaded yet. Train, register, then POST /reload.")
except Exception:
    st.sidebar.error("API not reachable")
drift = read_json("drift_summary.json")
if drift:
    label = "HIGH" if drift["drift_detected"] else "OK"
    st.sidebar.write("Drift:", label, f"({drift['share_drifted']:.0%} of columns)")
retrain = read_json("retrain_log.json")
if retrain:
    st.sidebar.write("Last retrain:", retrain["decision"])

tab_estimate, tab_market = st.tabs(["Estimate", "Market"])

# ---------------- tab 1: estimate ----------------
with tab_estimate:
    c1, c2, c3 = st.columns(3)
    brands = choices["brands"]
    brand = c1.selectbox("Brand", brands, index=brands.index("Toyota") if "Toyota" in brands else 0)
    car_model = c2.selectbox("Model", choices["models_by_brand"].get(brand, [brand]))
    city = c3.selectbox("City", choices["cities"])

    c4, c5, c6 = st.columns(3)
    year = c4.number_input("Model year", 1970, 2026, 2018)
    mileage = c5.number_input("Mileage (km)", 0, 1_000_000, 60000, step=1000)
    engine_cc = c6.number_input("Engine (cc)", 300, 8000, 1300, step=100)

    c7, c8 = st.columns(2)
    fuel = c7.selectbox("Fuel", choices["fuels"])
    transmission = c8.selectbox("Transmission", choices["transmissions"])

    if st.button("Estimate price", type="primary"):
        payload = dict(year=year, mileage=mileage, brand=brand, car_model=car_model, city=city,
                       engine_cc=engine_cc, fuel=fuel, transmission=transmission)
        try:
            r = requests.post(f"{API}/predict", json=payload, timeout=10)
            if r.ok:
                st.success(f"Estimated price: Rs {r.json()['predicted_price']:,.0f}")
            else:
                st.error(r.text)
        except Exception as e:
            st.error(f"API not reachable: {e}")

# ---------------- tab 2: market ----------------
with tab_market:
    eda = read_json("eda_summary.json")
    if eda:
        left, right = st.columns(2)
        left.subheader("Median price by model year")
        left.bar_chart(pd.Series(eda["median_price_by_year"]))
        right.subheader("Median price by brand")
        right.bar_chart(pd.Series(eda["median_price_by_brand"]))
    else:
        st.info("Run python -m src.make_batches to create the market summary.")
