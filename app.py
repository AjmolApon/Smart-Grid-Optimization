import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.append("src")
from forecasting import create_features, load_processed_data, train_xgboost_model
from optimizer import run_smart_grid_optimization

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Grid Energy Management System",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ Smart Grid Optimization & Energy Management System")
st.markdown(
    "**MSc Research Project:** Real-world Demand Forecasting (XGBoost) & Battery Scheduling (PuLP LP Solver)"
)

# -----------------------------------------------------------------------------
# Sidebar: Control Panel
# -----------------------------------------------------------------------------
st.sidebar.header("🎛️ Grid & Battery Parameters")

battery_capacity = st.sidebar.slider(
    "Battery Capacity (kWh)", 5.0, 50.0, 15.0, step=2.5
)
max_charge = st.sidebar.slider(
    "Max Charge Rate (kW)", 1.0, 10.0, 4.0, step=0.5
)
max_discharge = st.sidebar.slider(
    "Max Discharge Rate (kW)", 1.0, 10.0, 4.0, step=0.5
)
efficiency = (
    st.sidebar.slider("Battery Efficiency (%)", 70, 100, 90, step=5) / 100.0
)

forecast_hours = st.sidebar.selectbox(
    "Optimization Horizon (Hours)", [24, 48, 72, 168], index=0
)

# -----------------------------------------------------------------------------
# Main Dashboard Logic
# -----------------------------------------------------------------------------
st.info("⏳ Data Pipeline & Machine Learning Models initialize হচ্ছে...")


@st.cache_data
def get_optimized_data(hours):
    df_raw = load_processed_data("data/raw_data.csv")
    df_sample = df_raw.head(hours).copy()
    return df_sample


df_forecast = get_optimized_data(forecast_hours)

# অপ্টিমাইজেশন রান
df_opt, opt_cost = run_smart_grid_optimization(
    df_forecast,
    battery_capacity=battery_capacity,
    max_charge_rate=max_charge,
    max_discharge_rate=max_discharge,
    battery_efficiency=efficiency,
)

# বেসলাইন হিসাব (ব্যাটারি ছাড়া খরচ)
baseline_cost = 0
for _, row in df_forecast.iterrows():
    h = row["Timestamp"].hour
    p = 0.35 if 17 <= h <= 21 else 0.12
    net = max(0, row["Load_kW"] - row["Solar_kW"])
    baseline_cost += net * p

savings = baseline_cost - opt_cost
savings_pct = (savings / baseline_cost * 100) if baseline_cost > 0 else 0

# -----------------------------------------------------------------------------
# Key Metrics
# -----------------------------------------------------------------------------
st.subheader("📊 Performance & Cost Summary")
col1, col2, col3, col4 = st.columns(4)

col1.metric("Unoptimized Grid Cost", f"${baseline_cost:.2f}")
col2.metric(
    "Optimized Grid Cost", f"${opt_cost:.2f}", delta=f"-${savings:.2f}"
)
col3.metric("Cost Savings (%)", f"{savings_pct:.1f}%")
col4.metric("Total Load Demand", f"{df_forecast['Load_kW'].sum():.1f} kWh")

# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
st.subheader("📈 Energy Dispatch Schedule & Battery State of Charge")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Graph 1: Power Balance Profiles
ax1.plot(
    df_opt["Timestamp"],
    df_opt["Load_kW"],
    label="Load Demand (kW)",
    color="black",
    linestyle="--",
)
ax1.plot(
    df_opt["Timestamp"],
    df_opt["Solar_kW"],
    label="Solar Generation (kW)",
    color="orange",
)
ax1.plot(
    df_opt["Timestamp"],
    df_opt["Grid_Import_kW"],
    label="Grid Import (kW)",
    color="red",
    alpha=0.7,
)
ax1.bar(
    df_opt["Timestamp"],
    df_opt["Bat_Charge_kW"],
    width=0.03,
    label="Bat Charge (kW)",
    color="green",
    alpha=0.5,
)
ax1.bar(
    df_opt["Timestamp"],
    -df_opt["Bat_Discharge_kW"],
    width=0.03,
    label="Bat Discharge (kW)",
    color="blue",
    alpha=0.5,
)
ax1.set_ylabel("Power (kW)")
ax1.set_title("Grid Power Balance & Battery Charging Dynamic")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle=":", alpha=0.6)

# Graph 2: SoC Dynamic
ax2.plot(
    df_opt["Timestamp"],
    df_opt["SoC_kWh"],
    label="Battery SoC (kWh)",
    color="purple",
    linewidth=2,
)
ax2.set_ylabel("State of Charge (kWh)")
ax2.set_xlabel("Timestamp")
ax2.set_title("Battery SoC Dynamic over Time")
ax2.legend(loc="upper right")
ax2.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
st.pyplot(fig)

# Data Table
with st.expander("📄 View Raw Optimization Data Table"):
    st.dataframe(df_opt)