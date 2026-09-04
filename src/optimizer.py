import numpy as np
import pandas as pd
import pulp


def run_smart_grid_optimization(
    df_forecast,
    battery_capacity=10.0,  # kWh
    max_charge_rate=3.0,  # kW
    max_discharge_rate=3.0,  # kW
    battery_efficiency=0.9,  # 90% Efficiency
):
    """PuLP Linear Programming Optimizer for Battery Scheduling & Grid Cost Minimization."""
    T = len(df_forecast)  # সময় (২৪ ঘণ্টা বা নির্দিষ্ট সময়কাল)

    # ১. অপ্টিমাইজেশন প্রবলেম ডিক্লেয়ার করা (Minimization Objective)
    prob = pulp.LpProblem("Smart_Grid_Optimization", pulp.LpMinimize)

    # ২. ডিসিশন ভ্যারিয়েবলসমূহ (Decision Variables)
    grid_import = [
        pulp.LpVariable(f"Grid_Import_{t}", lowBound=0) for t in range(T)
    ]
    grid_export = [
        pulp.LpVariable(f"Grid_Export_{t}", lowBound=0) for t in range(T)
    ]
    bat_charge = [
        pulp.LpVariable(f"Bat_Charge_{t}", lowBound=0, upBound=max_charge_rate)
        for t in range(T)
    ]
    bat_discharge = [
        pulp.LpVariable(
            f"Bat_Discharge_{t}", lowBound=0, upBound=max_discharge_rate
        )
        for t in range(T)
    ]
    soc = [
        pulp.LpVariable(
            f"SoC_{t}",
            lowBound=battery_capacity * 0.1,  # Minimum 10% Reserve
            upBound=battery_capacity * 0.95,  # Maximum 95% SoC
        )
        for t in range(T)
    ]

    # ৩. অবজেক্টিভ ফাংশন: মোট গ্রিড খরচ কমানো
    # ToU Tariff Structure ($/kWh)
    # Peak Hours (17:00 - 21:00): $0.35/kWh
    # Off-Peak Hours (অন্য সময়): $0.12/kWh
    # Export Price (Feed-in Tariff): $0.05/kWh
    total_cost = []
    for t in range(T):
        hour = df_forecast.iloc[t]["Timestamp"].hour
        electricity_price = 0.35 if 17 <= hour <= 21 else 0.12
        feed_in_tariff = 0.05

        cost_t = (grid_import[t] * electricity_price) - (
            grid_export[t] * feed_in_tariff
        )
        total_cost.append(cost_t)

    prob += pulp.lpSum(total_cost), "Total_Energy_Cost"

    # ৪. কনস্ট্রেইন্টস (Constraints)
    for t in range(T):
        load = df_forecast.iloc[t]["Load_kW"]
        solar = df_forecast.iloc[t]["Solar_kW"]

        # A. পাওয়ার ব্যালেন্স ইকুয়েশন
        prob += (
            grid_import[t] + solar + bat_discharge[t]
            == load + bat_charge[t] + grid_export[t],
            f"Power_Balance_{t}",
        )

        # B. ব্যাটারি স্টেট অফ চার্জ (SoC) ডাইনামিক্স
        if t == 0:
            initial_soc = battery_capacity * 0.5  # ৫০% চার্জ দিয়ে শুরু
            prob += (
                soc[t]
                == initial_soc
                + (bat_charge[t] * battery_efficiency)
                - (bat_discharge[t] / battery_efficiency),
                f"SoC_Update_{t}",
            )
        else:
            prob += (
                soc[t]
                == soc[t - 1]
                + (bat_charge[t] * battery_efficiency)
                - (bat_discharge[t] / battery_efficiency),
                f"SoC_Update_{t}",
            )

    # ৫. প্রবলেম সলভ করা
    solver = pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    # ৬. ফলাফল সাজিয়ে রিটার্ন করা
    results = []
    for t in range(T):
        results.append(
            {
                "Timestamp": df_forecast.iloc[t]["Timestamp"],
                "Load_kW": df_forecast.iloc[t]["Load_kW"],
                "Solar_kW": df_forecast.iloc[t]["Solar_kW"],
                "Grid_Import_kW": pulp.value(grid_import[t]),
                "Grid_Export_kW": pulp.value(grid_export[t]),
                "Bat_Charge_kW": pulp.value(bat_charge[t]),
                "Bat_Discharge_kW": pulp.value(bat_discharge[t]),
                "SoC_kWh": pulp.value(soc[t]),
                "Electricity_Price": 0.35
                if 17 <= df_forecast.iloc[t]["Timestamp"].hour <= 21
                else 0.12,
            }
        )

    df_results = pd.DataFrame(results)
    optimized_cost = pulp.value(prob.objective)
    return df_results, optimized_cost


if __name__ == "__main__":
    print("⏳ অপ্টিমাইজেশন ইঞ্জিন টেস্ট করা হচ্ছে (২৪ ঘণ্টার ডেটা দিয়ে)...")

    # টেস্ট করার জন্য raw_data থেকে ২৪ ঘণ্টার একটি সিম্পল স্যাম্পল লোড
    df_raw = pd.read_csv("data/raw_data.csv")
    df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"])
    df_sample = df_raw.head(24).copy()

    df_results, cost = run_smart_grid_optimization(df_sample)

    print("\n" + "=" * 50)
    print(f"🎉 অপ্টিমাইজেশন সম্পন্ন! স্ট্যাটাস: {pulp.LpStatus[1]}")
    print(f"💰 ২৪ ঘণ্টায় আনুমানিক মোট বিদ্যুৎ খরচ: ${cost:.2f}")
    print("=" * 50)
    print("\n📊 প্রথম ৫ ঘণ্টার অপ্টিমাইজড শিডিউল:")
    print(
        df_results[
            [
                "Timestamp",
                "Load_kW",
                "Solar_kW",
                "Bat_Charge_kW",
                "Bat_Discharge_kW",
                "Grid_Import_kW",
                "SoC_kWh",
            ]
        ].head()
    )