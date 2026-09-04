import os
import numpy as np
import pandas as pd


def process_txt_dataset(input_file_path, output_csv_path):
    print("⏳ আসল ডেটাসেট প্রসেসিং শুরু হয়েছে...")

    if not os.path.exists(input_file_path):
        print(f"❌ এরর: '{input_file_path}' ফাইলটি পাওয়া যায়নি!")
        return

    # Semicolon (;) স্প্লিট করে ডাটা লোড করা
    try:
        df = pd.read_csv(
            input_file_path,
            sep=";",
            low_memory=False,
            na_values=["?"],
            skipinitialspace=True,
        )
        print(f"✅ ফাইল লোড সম্পন্ন! মোট রো: {len(df):,}")
    except Exception as e:
        print(f"❌ ফাইল লোডিং এরর: {e}")
        return

    # কলামের নামের আসেপাশের স্পেস মুছে ফেলা
    df.columns = df.columns.str.strip()

    print("⏳ তারিখ এবং সময় পার্স করা হচ্ছে...")
    # Date এবং Time একসাথে করে Timestamp কলাম তৈরি
    df["Timestamp"] = pd.to_datetime(
        df["Date"] + " " + df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce"
    )

    # Global_active_power কলামকে লোড হিসেবে নেওয়া
    df["Load_kW"] = pd.to_numeric(df["Global_active_power"], errors="coerce")

    # মিসিং মান বাদ দেওয়া
    df = df.dropna(subset=["Timestamp", "Load_kW"])

    print("⏳ ১ মিনিটের ডেটাকে প্রতি ঘণ্টার গড়ে রূপান্তর (Hourly Resampling) করা হচ্ছে...")
    df_hourly = df.set_index("Timestamp").resample("h")["Load_kW"].mean().reset_index()

    print("☀️ সোলার জেনারেশন প্রোফাইল যুক্ত করা হচ্ছে...")
    hours = df_hourly["Timestamp"].dt.hour
    max_load = df_hourly["Load_kW"].max()

    # সোলার পাওয়ারের ফিজিক্স মডেল (দিনের বেলা সোলার আউটপুট থাকবে)
    solar_base = np.maximum(0, np.sin((hours - 6) * np.pi / 12)) * (max_load * 0.75)
    solar_noise = np.random.normal(0, 0.1, size=len(df_hourly))
    df_hourly["Solar_kW"] = np.clip(solar_base + solar_noise, 0, None)

    # ফাইনাল ক্লিন ডাটা তৈরি
    final_df = df_hourly[["Timestamp", "Solar_kW", "Load_kW"]].dropna()

    # CSV সেভ করা
    final_df.to_csv(output_csv_path, index=False)

    print("\n" + "=" * 50)
    print(f"🎉 সাফল্য! আসল ডেটাসেট তৈরি হয়ে '{output_csv_path}' এ সেভ হয়েছে।")
    print(f"📈 মোট ঘণ্টার সংখ্যা (Total Hours): {len(final_df):,}")
    print("=" * 50)


if __name__ == "__main__":
    txt_path = "data/household_data.txt"
    csv_path = "data/raw_data.csv"
    process_txt_dataset(txt_path, csv_path)