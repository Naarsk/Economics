import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from Finance.Thesis.palette import palette


def plot_sentiment_vs_irr(years, sentiment, errors, irr, title,fund_number_name):
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
    plt.savefig(fr"C:\Users\leocr\Projects\Economics\Finance\Thesis\Latex\img\{fund_number_name}_Sentiment_Index_vs_PE_IRR.png")
    plt.show()


def plot_dual_axis(years, sentiment, errors, secondary_series, secondary_label, title, fund_number_name):
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # sentiment with shaded error area
    ax1.plot(years, sentiment, marker="o", color=palette["primary_red"], label="Sentiment Index")
    ax1.fill_between(years, sentiment - errors, sentiment + errors,
                     color=palette["primary_red"], alpha=0.2)
    ax1.set_ylabel("Sentiment Index", color=palette["primary_red"])
    ax1.tick_params(axis="y", labelcolor=palette["primary_red"])

    # secondary axis
    ax2 = ax1.twinx()
    ax2.plot(years, secondary_series, marker="s", color=palette["dark_gray"], label=secondary_label)
    ax2.set_ylabel(secondary_label, color=palette["dark_gray"])
    ax2.tick_params(axis="y", labelcolor=palette["dark_gray"])

    fig.suptitle(title)
    ax1.grid(True)
    fig.tight_layout()
    plt.savefig(fr"C:\Users\leocr\Projects\Economics\Finance\Thesis\Latex\img\{fund_number_name}_{title}.png")
    plt.show()


def plot_avg_outlook_by_year(df: pd.DataFrame,
                             filename: str,
                             start_year=2019,
                             end_year=2025,
                             output_dir=r"C:\Users\leocr\Projects\Economics\Finance\Thesis\Latex\img"):
    """
    Calculates average outlook_num by year (2019–2025), plots it with error bars,
    and saves to the given filename.

    Error bars are inversely proportional to sqrt(n), where n is the number of entries.
    Gaps if <3 entries, but line connects valid points.
    """

    # ✅ Ensure date is datetime
    df["date_dt"] = pd.to_datetime(df["report_date"], format="%Y%m%d", errors="coerce")
    df = df[df["date_dt"].notna()]

    # ✅ Extract year and filter range
    df["year"] = df["date_dt"].dt.year
    df = df[(df["year"] >= start_year) & (df["year"] <= end_year)]

    # ✅ Compute mean + count per year
    grouped = df.groupby("year")["outlook_num"].agg(["mean", "count"])

    # ✅ Only keep mean if count >= 3
    grouped.loc[grouped["count"] < 3, "mean"] = float("nan")

    # ✅ Reindex to full year range
    years = range(start_year, end_year + 1)
    avg_by_year = grouped["mean"].reindex(years)
    counts = grouped["count"].reindex(years)

    # ✅ Define error bars (inverse sqrt of n)
    errors = 0.5 / np.sqrt(counts.replace(0, np.nan))  # avoid /0

    # ✅ Plot
    plt.figure(figsize=(8, 5))

    # Scatter with error bars
    plt.errorbar(avg_by_year.index, avg_by_year.values,
                 yerr=errors,
                 fmt="o", color=palette["primary_red"], ecolor=palette["dark_gray"], capsize=5)

    # Line only for valid points
    valid = avg_by_year.dropna()
    plt.plot(valid.index, valid.values, "-", color=palette["primary_red"])

    plt.title(f"Average Outlook by Year ({start_year}–{end_year})")
    plt.xlabel("Year")
    plt.ylabel("Average Outlook (1=Increase, 0=Stable, -1=Decrease)")
    plt.grid(True)

    # ✅ Save to file
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {filepath}")
