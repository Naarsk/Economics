import pandas as pd
import numpy as np

# 1. Load and aggregate GP cashflows
def load_cashflows(filepath: str) -> pd.DataFrame:
    df = pd.read_excel(filepath, parse_dates=['TRANSACTION DATE'])

    return df

# Define function to safely compute denominator
def safe_divide(numerator, denominator, eps=1e-6):
    return numerator / np.where(np.abs(denominator) < eps, np.nan, denominator)

def aggregate_cashflows(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime format
    df = df.copy()
    df["Quarter"] = df["TRANSACTION DATE"].dt.to_period("Q").dt.to_timestamp()

    # Ensure numeric types
    numeric_cols = ["TRANSACTION AMOUNT", "CUMULATIVE CONTRIBUTION", "CUMULATIVE DISTRIBUTION", "NET CASHFLOW"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    # Precompute Capital Calls and Distributions flags
    df["is_dist"] = df["TRANSACTION TYPE"] == "Distribution"
    df["is_call"] = df["TRANSACTION TYPE"] == "Capital Call"

    # Grouped aggregations
    grouped = df.groupby(["FUND ID", "Quarter"])

    agg_df = grouped.agg(
        NetCashflow=("NET CASHFLOW", "sum"),
        CumulativeContribution=("CUMULATIVE CONTRIBUTION", "max"),
        CumulativeDistribution=("CUMULATIVE DISTRIBUTION", "max"),
        DistAmount=("TRANSACTION AMOUNT", lambda x: x[df.loc[x.index, "is_dist"]].sum()),
        CallAmount=("TRANSACTION AMOUNT", lambda x: x[df.loc[x.index, "is_call"]].sum()),
    ).reset_index()

    # Compute derived metrics
    agg_df["NetInvestedCapital"] = agg_df["CumulativeContribution"] - agg_df["CumulativeDistribution"]
    agg_df["DistYield"] = safe_divide(agg_df["DistAmount"], agg_df["NetInvestedCapital"])
    agg_df["CapCallRate"] = safe_divide(np.abs(agg_df["CallAmount"]), agg_df["NetInvestedCapital"])
    agg_df["NetMultiple"] = safe_divide(agg_df["CumulativeDistribution"], agg_df["CumulativeContribution"])
    agg_df["NetFlowRatio"] = safe_divide(agg_df["NetCashflow"], agg_df["NetInvestedCapital"])

    # Optional: clean up columns
    final_cols = [
        "FUND ID", "Quarter", "NetCashflow", "NetInvestedCapital",
        "DistYield", "CapCallRate", "NetMultiple", "NetFlowRatio"
    ]

    return agg_df[final_cols]

# 2. Generate synthetic S&P 500 returns
def generate_sp500_returns(quarters: pd.Series) -> pd.DataFrame:
    np.random.seed(1)
    returns = np.random.normal(loc=0.02, scale=0.05, size=len(quarters))
    return pd.DataFrame({'Quarter': quarters, 'Rm': returns})

# 3. Generate synthetic risk-free rates
def generate_risk_free_rate(quarters: pd.Series) -> pd.DataFrame:
    np.random.seed(2)
    rates = np.random.normal(loc=0.01, scale=0.005, size=len(quarters))
    return pd.DataFrame({'Quarter': quarters, 'Rf': rates})

# 4. Generate artificial sentiment index
def generate_sentiment_index(fund_ids: list, quarters: pd.Series) -> pd.DataFrame:
    np.random.seed(42)
    idx = pd.MultiIndex.from_product([fund_ids, quarters], names=['FUND ID', 'Quarter'])
    sentiment = pd.DataFrame({
        'Sentiment': np.random.normal(0, 1, len(idx))
    }, index=idx).reset_index()
    return sentiment

# 5. Merge everything into a panel dataset
def build_panel_dataset(cashflow_panel, sentiment, sp500_returns, rfr):
    panel = cashflow_panel.merge(sentiment, on=['FUND ID', 'Quarter'], how='left')
    market = sp500_returns.merge(rfr, on='Quarter', how='inner')
    panel = panel.merge(market, on='Quarter', how='left')
    return panel


# Usage example
filepath = 'files/GP_cashflows.xlsx'
cashflows = load_cashflows(filepath)
cashflow_panel = aggregate_cashflows(cashflows)

quarters = pd.period_range(cashflow_panel['Quarter'].min(), cashflow_panel['Quarter'].max(), freq='Q').to_timestamp()
fund_ids = cashflow_panel['FUND ID'].unique().tolist()

# Generate synthetic data
sp500_returns = generate_sp500_returns(quarters)
rfr = generate_risk_free_rate(quarters)
sentiment = generate_sentiment_index(fund_ids, quarters)

# Build dataset
panel_data = build_panel_dataset(cashflow_panel, sentiment, sp500_returns, rfr)

# Save or display
panel_data.to_csv("panel_dataset.csv", index=False)
print(panel_data.head(10))
