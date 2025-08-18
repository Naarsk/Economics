from Finance.Thesis.B_analysis_functions import load_sentiment, load_pe_irr
from Finance.Thesis.C_plotting_functions import plot_sentiment_vs_irr

# --- Parameters ---
start_year, end_year = 2018, 2025
years = range(start_year, end_year + 1)
fund_number_name = "22_TCG"

# --- Main ---
sentiment, errors = load_sentiment(
    rf"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel\{fund_number_name}.xlsx",
    start_year, end_year
)

pe_irr = load_pe_irr(
    r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\Excels\Horizon_IRR.xlsx",
    years
)

plot_sentiment_vs_irr(years, sentiment, errors, pe_irr, "Sentiment Index vs. Private Equity IRR", fund_number_name=fund_number_name)
