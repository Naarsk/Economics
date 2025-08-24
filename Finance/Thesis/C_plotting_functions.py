import os

import pandas as pd
from matplotlib import pyplot as plt

from Finance.Thesis.B_analysis_functions import load_sentiment
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


def plot_avg_outlook(path,
                     filename: str,
                     start_year=2019,
                     end_year=2025,
                     freq ="yearly",
                     output_dir=r"C:\Users\leocr\Projects\Economics\Finance\Thesis\Latex\img"):
    """
    Calculates average outlook_num by year or quarter (2019–2025), plots it with error bars,
    and saves to the given filename.

    Error bars are inversely proportional to sqrt(n), where n is the number of entries.
    Gaps if <3 entries, but line connects valid points.
    """

    grouped = load_sentiment(path,start_year,end_year, freq)
    avg = grouped["sentiment"]
    errors = grouped["errors"]

    # ✅ Plot
    plt.figure(figsize=(10, 5))

    plt.errorbar(avg.index, avg.values,
                 yerr=errors,
                 fmt="o", color=palette["primary_red"], ecolor=palette["dark_gray"], capsize=5)

    valid = avg.dropna()
    plt.plot(valid.index, valid.values, "-", color=palette["primary_red"])

    title_freq = "Year" if freq == "yearly" else "Quarter"
    plt.title(f"Average Outlook by {title_freq} ({start_year}–{end_year})")
    plt.xlabel(title_freq)
    plt.ylabel("Average Outlook (1=Increase, 0=Stable, -1=Decrease)")
    plt.grid(True)

    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Plot saved to {filepath}")



def plot_distributions(path, fund_code):
    """Plots histogram distributions for outlook_num and confidence."""
    df = pd.read_excel(path)

    # Convert confidence to numeric, ignoring errors
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    plt.figure(figsize=(12, 5))

    # ✅ Outlook_num distribution
    plt.subplot(1, 2, 1)
    if "outlook_num" in df.columns:
        df["outlook_num"].dropna().hist(bins=3, rwidth=0.8)
        plt.xticks([-1, 0, 1], ["Decrease (-1)", "Stable (0)", "Increase (1)"])
        plt.title("Distribution of Outlook")
        plt.xlabel("Outlook")
        plt.ylabel("Count")

    # ✅ Confidence distribution
    plt.subplot(1, 2, 2)
    if "confidence" in df.columns:
        df["confidence"].dropna().hist(bins=10, rwidth=0.8)
        plt.title("Distribution of Confidence")
        plt.xlabel("Confidence")
        plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(f"Latex/img/{fund_code}_outlook_distribution.png")
    plt.show()
    plt.close()
