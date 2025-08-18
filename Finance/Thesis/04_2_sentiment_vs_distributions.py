from Finance.Thesis.B_analysis_functions import load_sentiment, load_gp_cashflows
from Finance.Thesis.C_plotting_functions import plot_dual_axis

fund_number_name = "21_TPG"
fund_name = "TPG"

# -----------------------------
# Main
# -----------------------------
sentiment_path = rf"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel\{fund_number_name}.xlsx"
gp_path = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\GP_cashflows.xlsx"

sentiment, errors = load_sentiment(sentiment_path, start_year=2017, end_year=2025)
net_cf, dpi = load_gp_cashflows(gp_path, fund=fund_name)

# -----------------------------
# Align with 1-year lag in sentiment
# -----------------------------
sentiment_lagged = sentiment.shift(-1)  # sentiment at t → used for year t+1
errors_lagged = errors.shift(-1)

years = sentiment.index
net_cf = net_cf.reindex(years)
dpi = dpi.reindex(years)

# plots
plot_dual_axis(years, sentiment_lagged, errors_lagged, net_cf, "Net Cashflow", f"Lagged Sentiment vs Net Cashflow ({fund_name})", fund_number_name=fund_number_name)
plot_dual_axis(years, sentiment_lagged, errors_lagged, dpi, "DPI", f"Lagged Sentiment vs DPI ({fund_name})", fund_number_name=fund_number_name)
