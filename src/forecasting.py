import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb


def load_processed_data(file_path="data/raw_data.csv"):
    """প্রসেস করা রিয়েল ডেটাসেট লোড করে।"""
    df = pd.read_csv(file_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    return df


def create_features(df):
    """টাইম-সিরিজ ডেটা থেকে ফিচার ইঞ্জিনিয়ারিং করে।"""
    df = df.copy()
    df["Hour"] = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Month"] = df["Timestamp"].dt.month

    # Lag features (আগের ১ ঘণ্টা ও ২৪ ঘণ্টার ডেটা)
    df["Solar_Lag1"] = df["Solar_kW"].shift(1)
    df["Solar_Lag24"] = df["Solar_kW"].shift(24)
    df["Load_Lag1"] = df["Load_kW"].shift(1)
    df["Load_Lag24"] = df["Load_kW"].shift(24)

    return df.dropna()


def train_xgboost_model(X_train, y_train):
    """XGBoost Regressor ট্রেন করে।"""
    model = xgb.XGBRegressor(
        n_estimators=150, learning_rate=0.03, max_depth=6, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def main():
    print("⏳ প্রসেস করা আসল ডেটা লোড করা হচ্ছে...")
    df = load_processed_data()
    df_featured = create_features(df)

    features = [
        "Hour",
        "DayOfWeek",
        "Month",
        "Solar_Lag1",
        "Solar_Lag24",
        "Load_Lag1",
        "Load_Lag24",
    ]
    X = df_featured[features]

    # ১. সোলার মডেল ট্রেনিং (Solar Forecasting Model)
    y_solar = df_featured["Solar_kW"]
    X_train, X_test, y_solar_train, y_solar_test = train_test_split(
        X, y_solar, test_size=0.2, shuffle=False
    )

    print("☀️ সোলার ফোরকাস্টিং মডেল ট্রেনিং চলছে...")
    solar_model = train_xgboost_model(X_train, y_solar_train)
    solar_preds = solar_model.predict(X_test)
    solar_rmse = np.sqrt(mean_squared_error(y_solar_test, solar_preds))
    solar_mae = mean_absolute_error(y_solar_test, solar_preds)
    print(
        f"  ➡️ Solar Model Metrics - RMSE: {solar_rmse:.3f} kW | MAE: {solar_mae:.3f} kW"
    )

    # ২. লোড মডেল ট্রেনিং (Load Demand Forecasting Model)
    y_load = df_featured["Load_kW"]
    _, _, y_load_train, y_load_test = train_test_split(
        X, y_load, test_size=0.2, shuffle=False
    )

    print("⚡ লোড ডিমান্ড ফোরকাস্টিং মডেল ট্রেনিং চলছে...")
    load_model = train_xgboost_model(X_train, y_load_train)
    load_preds = load_model.predict(X_test)
    load_rmse = np.sqrt(mean_squared_error(y_load_test, load_preds))
    load_mae = mean_absolute_error(y_load_test, load_preds)
    print(
        f"  ➡️ Load Model Metrics - RMSE: {load_rmse:.3f} kW | MAE: {load_mae:.3f} kW"
    )

    # মডেল সেভ করা
    joblib.dump(solar_model, "models/solar_model.pkl")
    joblib.dump(load_model, "models/load_model.pkl")
    print("\n✅ সফলভাবে আসল ডেটায় প্রশিক্ষিত মডেল দুটি 'models/' ফোল্ডারে সেভ হয়েছে!")


if __name__ == "__main__":
    main()