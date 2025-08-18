import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# -----------------------------
# Load Sentiment
# -----------------------------
def load_sentiment(sentiment_path, start_year=2019, end_year=2025):
    df = pd.read_excel(sentiment_path)
    df["date_dt"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
    df = df[df["date_dt"].notna()]

    df["year"] = df["date_dt"].dt.year
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    grouped = df.groupby("year")["outlook_num"].agg(["mean", "count"])
    years = range(start_year, end_year + 1)
    grouped = grouped.reindex(years)

    errors = 1 / np.sqrt(grouped["count"].replace(0, np.nan))
    return grouped["mean"], errors


# -----------------------------
# Load GP Cashflows
# -----------------------------
def load_gp_cashflows(filepath):
    df = pd.read_excel(filepath)
    df = df[(df["TRANSACTION TYPE"] == "Distribution") & (df["FUND MANAGER"] == "TPG")]

    df["date_dt"] = pd.to_datetime(df["TRANSACTION DATE"], errors="coerce")
    df["year"] = df["date_dt"].dt.year

    net_cf = df.groupby("year")["NET CASHFLOW"].sum()
    ratio = (df["NET CASHFLOW"] / df["CUMULATIVE CONTRIBUTION"]).groupby(df["year"]).sum()
    return net_cf, ratio


# -----------------------------
# Weighted Regression
# -----------------------------
def weighted_regression(y, x, weights):
    mask = (~y.isna()) & (~x.isna()) & (~weights.isna())
    y, x, w = y[mask], x[mask], weights[mask]

    X = sm.add_constant(x)
    model = sm.WLS(y, X, weights=w)
    results = model.fit()
    return results


# -----------------------------
# Plot helper
# -----------------------------
def plot_dual_axis(years, sentiment, errors, secondary_series, secondary_label, secondary_color, title):
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # sentiment with shaded error area
    ax1.plot(years, sentiment, marker="o", color="#C00000", label="Sentiment Index")
    ax1.fill_between(years, sentiment - errors, sentiment + errors,
                     color="#C00000", alpha=0.2)
    ax1.set_ylabel("Sentiment Index", color="#C00000")
    ax1.tick_params(axis="y", labelcolor="#C00000")

    # secondary axis
    ax2 = ax1.twinx()
    ax2.plot(years, secondary_series, marker="s", color=secondary_color, label=secondary_label)
    ax2.set_ylabel(secondary_label, color=secondary_color)
    ax2.tick_params(axis="y", labelcolor=secondary_color)

    fig.suptitle(title)
    ax1.grid(True)
    fig.tight_layout()
    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    sentiment_path = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel\21_TPG.xlsx"
    gp_path = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\GP_cashflows.xlsx"

    sentiment, errors = load_sentiment(sentiment_path)
    net_cf, ratio = load_gp_cashflows(gp_path)

    # -----------------------------
    # Align with 1-year lag in sentiment
    # -----------------------------
    sentiment_lagged = sentiment.shift(-1)  # sentiment at t → used for year t+1
    errors_lagged = errors.shift(-1)

    years = sentiment.index
    net_cf = net_cf.reindex(years)
    ratio = ratio.reindex(years)

    # regressions (with lagged sentiment)
    res_cf = weighted_regression(net_cf, sentiment_lagged, 1 / errors_lagged ** 2)
    res_ratio = weighted_regression(ratio, sentiment_lagged, 1 / errors_lagged ** 2)

    print("Regression 1: NET CASHFLOW ~ Lagged Sentiment")
    print(res_cf.summary())
    print("\nRegression 2: NET CF / CUMULATIVE CONTRIBUTION ~ Lagged Sentiment")
    print(res_ratio.summary())

    # plots
    plot_dual_axis(years, sentiment_lagged, errors_lagged, net_cf, "Net Cashflow", "blue",
                   "Lagged Sentiment vs Net Cashflow (TPG)")
    plot_dual_axis(years, sentiment_lagged, errors_lagged, ratio, "Net CF / Cumulative Contribution", "green",
                   "Lagged Sentiment vs Net CF / Cumulative Contribution (TPG)")
