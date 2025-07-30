import os
import json
import pandas as pd

json_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\parsed_json"
excel_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel"
excel_path = os.path.join(excel_dir, "clean_outlooks.xlsx")

data = []

def clean_outlooks_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the outlooks DataFrame: keep required columns, normalize date,
    and add numeric outlook column."""

    # ✅ keep only required columns (ignore missing ones safely)
    required_cols = ["name", "date", "outlook", "magnitude", "confidence", "explanation", "source_file"]
    df = df[[col for col in required_cols if col in df.columns]].copy()

    # ✅ convert date to yyyymmdd
    def normalize_date(val):
        try:
            return pd.to_datetime(str(val), errors="coerce").strftime("%Y%m%d")
        except Exception:
            return None

    df["date"] = df["date"].apply(normalize_date)

    # ✅ drop rows with invalid dates
    df = df[df["date"].notna()]

    # ✅ map outlook to numeric values
    outlook_map = {"increase": 1, "stable": 0, "decrease": -1}
    df["outlook_num"] = df["outlook"].map(outlook_map)

    # ✅ drop rows where outlook is not one of the expected values
    df = df[df["outlook_num"].notna()]

    return df

for file in os.listdir(json_dir):
    print("Processing {}".format(file))
    if file.endswith("_parsed.json"):
        filepath = os.path.join(json_dir, file)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                results = json_data.get("results", None)

                # ✅ only keep entries where results is a dict
                if isinstance(results, dict):
                    results["source_file"] = file
                    data.append(results)
                else:
                    print(f"[SKIP] {file} -> results is not a dict")

        except Exception as e:
            print(f"[WARNING] Could not parse {file}: {e}")

# ✅ Create DataFrame from list of dicts
df = pd.DataFrame(data)

clean_df = clean_outlooks_df(df)

# ✅ Save to Excel
os.makedirs(excel_dir, exist_ok=True)
clean_df.to_excel(excel_path, index=False)
print(f"[INFO] Saved {len(df)} entries to {excel_path}")
