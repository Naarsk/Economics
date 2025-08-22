import json
import os
import numpy as np
import pandas as pd
from statsmodels import api as sm


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


def load_sentiment(path, start_year, end_year, freq="yearly", lag=0) -> pd.DataFrame:
    """
    Load and aggregate sentiment by year, semiannual, or quarter, with optional lagging.

    Parameters
    ----------
    path : str
        Path to Excel file with 'report_date' and 'outlook_num'.
    start_year, end_year : int
        Start and end year for filtering.
    freq : {"yearly", "semiannual", "quarterly"}
        Aggregation frequency.
    lag : int, optional (default=0)
        Number of periods to shift sentiment forward (+) or backward (-).
        Example: lag=1 means sentiment at t is shifted to affect t+1.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'sentiment': average sentiment
        - 'count': number of observations
        - 'errors': standard error proxy
        - 'sentiment_lagged': sentiment shifted by lag (if lag != 0)
    """
    # --- Load data ---
    df = pd.read_excel(path)
    df["date_dt"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
    df = df[df["date_dt"].notna()]

    df["year"] = df["date_dt"].dt.year
    df = df[(df["year"] >= start_year - lag) & (df["year"] <= end_year)]

    # --- Define period based on frequency ---
    if freq == "yearly":
        df["period"] = df["year"]

    elif freq == "quarterly":
        df["quarter"] = df["date_dt"].dt.quarter
        df["period"] = df["year"] + (df["quarter"] - 1) / 4

    elif freq == "semiannual":
        df["half"] = np.where(df["date_dt"].dt.month <= 6, 0.0, 0.5)
        df["period"] = df["year"] + df["half"]

    else:
        raise ValueError("freq must be 'yearly', 'semiannual', or 'quarterly'")

    # --- Aggregate sentiment ---
    out = df.groupby("period")["outlook_num"].agg(["mean", "count"])
    out.rename(columns={"mean": "sentiment"}, inplace=True)
    out["errors"] = 0.5 / np.sqrt(out["count"].replace(0, np.nan))

    # --- Apply lag if needed ---
    if lag != 0:
        out["sentiment_lagged"] = out["sentiment"].shift(lag)
        out["errors_lagged"] = out["errors"].shift(lag)

    return out.dropna()


def load_pe_irr(path, start_year, end_year):
    """
    Load PE IRR Excel file and return annual averages aligned with years.

    Parameters
    ----------
    path : str
        Path to Excel file.
    start_year, end_year : int
        Start and end year for filtering.

    Returns
    -------
    pd.DataFrame
        DataFrame with index named 'period' (years) and column 'pe_irr'.
    """
    years = range(start_year, end_year + 1)

    # Load file
    df = pd.read_excel(path)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])
    df = df.set_index(df.columns[0])

    # Take the first column (PE 1-year IRR) and convert % to decimal
    pe_1yr_irr = df.iloc[:, 0] / 100.0

    # Group by year and take mean
    pe_annual = pe_1yr_irr.groupby(pe_1yr_irr.index.year).mean()

    # Convert to DataFrame
    pe_annual = pe_annual.reindex(years).to_frame(name="pe_irr")
    pe_annual.index.name = "period"

    return pe_annual


def load_gp_cashflows(cashflow_path, fund, start_year=None, end_year=None, freq="yearly") -> pd.DataFrame:
    """
    Load GP cashflows and aggregate by year, semiannual, or quarter.

    Parameters
    ----------
    cashflow_path : str
        Path to Excel file with GP cashflows.
    fund : str
        Fund manager name to filter.
    start_year, end_year : int, optional
        Restrict range of years.
    freq : {"yearly", "semiannual", "quarterly"}
        Aggregation frequency.

    Returns
    -------
    pd.DataFrame
        DataFrame with net_cf, dpi, abs_change_net_cf, rel_change_net_cf
    """
    df = pd.read_excel(cashflow_path)
    df = df[(df["TRANSACTION TYPE"] == "Distribution") & (df["FUND MANAGER"] == fund)]

    df["date_dt"] = pd.to_datetime(df["TRANSACTION DATE"], errors="coerce")
    df = df[df["date_dt"].notna()]

    df["year"] = df["date_dt"].dt.year
    if start_year and end_year:
        df = df[(df["year"] >= start_year-1) & (df["year"] <= end_year)]

    # --- Define period based on frequency ---
    if freq == "yearly":
        df["period"] = df["year"]
    elif freq == "quarterly":
        df["quarter"] = df["date_dt"].dt.quarter
        df["period"] = df["year"] + (df["quarter"] - 1) / 4
    elif freq == "semiannual":
        df["half"] = df["date_dt"].dt.month.apply(lambda m: 1 if m <= 6 else 2)
        df["period"] = df["year"] + (df["half"] - 1) / 2
    else:
        raise ValueError("freq must be 'yearly', 'semiannual', or 'quarterly'")

    # --- Aggregate ---
    amount = df["TRANSACTION AMOUNT"].groupby(df["period"]).sum()
    net_cf = df["NET CASHFLOW"].groupby(df["period"]).sum()
    dpi = (-df["CUMULATIVE DISTRIBUTION"] / df["CUMULATIVE CONTRIBUTION"]).groupby(df["period"]).mean()

    # --- Combine into DataFrame ---
    out = pd.concat([amount.rename("amount"), net_cf.rename("net_cf"), dpi.rename("dpi")], axis=1)

    # --- Absolute + relative changes ---
    out["abs_change_net_cf"] = out["net_cf"].diff()
    out["rel_change_net_cf"] = out["net_cf"].diff() / out["net_cf"].shift(1).abs()
    out["abs_change_amount"] = out["amount"].diff()
    out["rel_change_amount"] = out["amount"].diff() / out["amount"].shift(1).abs()

    return out.dropna()


def load_snp500(path, start_year=None, end_year=None, freq="yearly") -> pd.DataFrame:
    """
    Load S&P500 close prices and compute log returns aggregated
    to quarterly, semiannual, or yearly frequency.

    Parameters
    ----------
    path : str
        Path to Excel file with S&P500 data.
    start_year, end_year : int, optional
        Start and end year for filtering.
    freq : {"yearly", "quarterly", "semiannual"}
        Aggregation frequency.

    Returns
    -------
    pd.DataFrame
        DataFrame with index named 'period' and column 'snp500'.
    """
    # --- Load daily S&P500 ---
    sp500_df = pd.read_excel(path)
    sp500_df["Date"] = pd.to_datetime(sp500_df["Date"])
    sp500_df = sp500_df.set_index("Date")

    # --- Map frequency to pandas offset ---
    freq_map = {
        "yearly": "YE",
        "quarterly": "Q",
        "semiannual": "2Q",
    }
    if freq not in freq_map:
        raise ValueError(f"freq must be one of {list(freq_map.keys())}")

    # --- Resample to selected frequency ---
    sp500_period = sp500_df["Close"].resample(freq_map[freq]).last()

    # --- Compute log returns ---
    log_ret = np.log(sp500_period / sp500_period.shift(1)).dropna()

    # --- Apply year filtering ---
    if start_year:
        log_ret = log_ret[log_ret.index.year >= start_year]
    if end_year:
        log_ret = log_ret[log_ret.index.year <= end_year]

    # --- Build custom index ---
    if freq == "yearly":
        new_index = log_ret.index.year
    elif freq == "quarterly":
        new_index = [y + (q - 1) / 4 for y, q in zip(log_ret.index.year, log_ret.index.quarter)]
    elif freq == "semiannual":
        new_index = [y + 0.5 * (1 if m > 6 else 0) for y, m in zip(log_ret.index.year, log_ret.index.month)]

    # Return DataFrame with proper naming
    out = log_ret.copy()
    out.index = new_index
    out.index.name = "period"
    out = out.to_frame(name="snp500")

    return out


def interpolate_sentiment_dpi(sentiment_path, cashflow_path, fund, start_year, end_year):
    """Load and aggregate sentiment by year, returning mean, count, and error bands."""
    df = pd.read_excel(sentiment_path)
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


def weighted_regression(y, X, weights):
    # Ensure X is a DataFrame so multiple regressors work
    if isinstance(X, pd.Series):
        X = X.to_frame()

    # Drop NaNs consistently across y, X, and weights
    mask = (~y.isna()) & (~weights.isna())
    for col in X.columns:
        mask &= ~X[col].isna()

    y, X, w = y[mask], X[mask], weights[mask]

    # Add constant for intercept
    X = sm.add_constant(X)

    model = sm.WLS(y, X, weights=w)
    results = model.fit()
    return results
