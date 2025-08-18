import matplotlib.pyplot as plt

from Finance.Thesis.functions import load_sentiment, load_pe_irr
from Finance.Thesis.palette import palette

# --- Parameters ---
start_year, end_year = 2015, 2025
years = range(start_year, end_year + 1)


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
