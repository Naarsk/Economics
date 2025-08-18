import json
import os

import numpy as np
import pandas as pd


def make_excel_from_json(json_dir, excel_dir, excel_name="clean_outlooks.xlsx"):
    excel_path = os.path.join(excel_dir, excel_name)

    data = []
    for file in os.listdir(json_dir):
        print("Processing {}".format(file))
        if file.endswith("_parsed.json"):
            filepath = os.path.join(json_dir, file)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                    results = json_data.get("results", None)

                    if isinstance(results, dict):
                        # ✅ single dict case
                        results["source_file"] = file
                        data.append(results)

                    elif isinstance(results, list):
                        # ✅ list of dicts case
                        for entry in results:
                            if isinstance(entry, dict):
                                entry["source_file"] = file
                                data.append(entry)

                    else:
                        print(f"[SKIP] {file} -> results not dict or list")

            except Exception as e:
                print(f"[WARNING] Could not parse {file}: {e}")

    # ✅ Create DataFrame from list of dicts
    df = pd.DataFrame(data)

    # Apply cleaning step if you have it
    if "clean_outlooks_df" in globals():
        clean_df = clean_outlooks_df(df)
    else:
        clean_df = df

    # ✅ Save to Excel
    os.makedirs(excel_dir, exist_ok=True)
    clean_df.to_excel(excel_path, index=False)

    print(f"Saved {len(clean_df)} rows to {excel_path}")


def clean_outlooks_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the outlooks DataFrame: keep required columns, normalize date,
    and add numeric outlook column."""

    # ✅ keep only required columns (ignore missing ones safely)
    required_cols = ["fund_name", "report_date", "outlook", "magnitude", "confidence", "explanation", "source_file"]
    df = df[[col for col in required_cols if col in df.columns]].copy()

    # ✅ convert date to yyyymmdd
    def normalize_date(val):
        try:
            return pd.to_datetime(str(val), errors="coerce").strftime("%Y%m%d")
        except Exception:
            return None

    df["report_date"] = df["report_date"].apply(normalize_date)

    # ✅ drop rows with invalid dates
    df = df[df["report_date"].notna()]

    # Ensure outlook is a string (take the first element if it's a list)
    df["outlook"] = df["outlook"].apply(lambda x: x[0] if isinstance(x, list) and x else x)

    # ✅ map outlook to numeric values
    outlook_map = {"increase": 1, "stable": 0, "decrease": -1}
    df["outlook_num"] = df["outlook"].map(outlook_map)

    # ✅ drop rows where outlook is not one of the expected values
    df = df[df["outlook_num"].notna()]

    return df


def load_sentiment(path, start_year, end_year):
    """Load and aggregate sentiment by year, returning mean, count, and error bands."""
    df = pd.read_excel(path)
    df["date_dt"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
    df = df[df["date_dt"].notna()]

    df["year"] = df["date_dt"].dt.year
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    grouped = df.groupby("year")["outlook_num"].agg(["mean", "count"])
    mean = grouped["mean"].reindex(range(start_year, end_year + 1))
    counts = grouped["count"].reindex(range(start_year, end_year + 1))

    # Errors = 1/sqrt(n), shaded area
    errors = 0.5 / np.sqrt(counts.replace(0, np.nan))
    return mean, errors


def load_pe_irr(path, years):
    """Load PE IRR Excel, return annual averages aligned with years."""
    df = pd.read_excel(path)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0])
    pe_1yr_irr = df.iloc[:, 0] / 100  # convert % to decimal if needed

    pe_annual = pe_1yr_irr.groupby(pe_1yr_irr.index.year).mean()
    return pe_annual.reindex(years)


def load_gp_cashflows(filepath, fund):
    df = pd.read_excel(filepath)
    df = df[(df["TRANSACTION TYPE"] == "Distribution") & (df["FUND MANAGER"] == fund)]

    df["date_dt"] = pd.to_datetime(df["TRANSACTION DATE"], errors="coerce")
    df["year"] = df["date_dt"].dt.year

    net_cf = df.groupby("year")["NET CASHFLOW"].sum()
    dpi = (df["CUMULATIVE DISTRIBUTION"] / df["CUMULATIVE CONTRIBUTION"]).groupby(df["year"]).mean()
    return net_cf, dpi



