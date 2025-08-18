import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
start_year, end_year = 2015, 2025
years = range(start_year, end_year + 1)

palette = {
    "primary_red": "#C00000",
    "dark_gray": "#4D4D4D",
    "soft_gray": "#A6A6A6",
    "accent_orange": "#E07B39",
    "deep_blue": "#003366",
    "muted_green": "#3A7D44"
}

# --- Functions ---
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


def plot_sentiment_vs_irr(years, sentiment, errors, irr, title):
    """Dual-axis plot with sentiment + shaded error and IRR."""
    fig, ax1 = plt.subplots(figsize=(9, 6))

    # Sentiment with shaded error band
    ax1.plot(years, sentiment, "o-", color=palette["primary_red"], label="Sentiment Index")
    ax1.fill_between(
        years,
        sentiment - errors,
        sentiment + errors,
        color=palette["primary_red"],
        alpha=0.2,
        label="Sentiment ± error"
    )
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Sentiment Index (avg)", color=palette["primary_red"])
    ax1.tick_params(axis="y", labelcolor=palette["primary_red"])
    ax1.set_xticks(list(years))
    ax1.legend( loc="upper left")
    # IRR on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(years, irr, "s--", color=palette["dark_gray"], label="PE 1yr IRR")
    ax2.set_ylabel("PE 1yr IRR", color=palette["dark_gray"])
    ax2.tick_params(axis="y", labelcolor=palette["dark_gray"])
    ax2.legend()

    # Title + layout
    plt.title(title)
    fig.tight_layout()
    plt.savefig(r"C:\Users\leocr\Projects\Economics\Finance\Thesis\Latex\img\21_TPG_Sentiment_Index_vs_PE_IRR.png")
    plt.show()


# --- Main ---
sentiment, errors = load_sentiment(
    r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel\21_TPG.xlsx",
    start_year, end_year
)

pe_irr = load_pe_irr(
    r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\Excels\Horizon_IRR.xlsx",
    years
)

plot_sentiment_vs_irr(years, sentiment, errors, pe_irr, "TPG Sentiment Index vs. Private Equity IRR")
