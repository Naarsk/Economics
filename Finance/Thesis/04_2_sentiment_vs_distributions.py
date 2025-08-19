from Finance.Thesis.B_analysis_functions import load_sentiment, load_gp_cashflows
from Finance.Thesis.C_plotting_functions import plot_dual_axis

fund_number_name = "21_TPG"
fund_name = "TPG"
start_year, end_year = 2018, 2025

# -----------------------------
# Main
# -----------------------------
sentiment_path = rf"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel\{fund_number_name}.xlsx"
gp_path = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\GP_cashflows.xlsx"

sentiment_df = load_sentiment(sentiment_path, start_year, end_year, freq='yearly')
sentiment =sentiment_df["mean"]
errors = sentiment_df["errors"]

cashflow_df = load_gp_cashflows(gp_path, fund=fund_name, start_year= start_year, end_year= end_year,freq='yearly')
net_cf = cashflow_df["net_cf"]
dpi = cashflow_df["dpi"]
abs_change_net_cf=cashflow_df["abs_change_net_cf"]
rel_change_net_cf =cashflow_df["rel_change_net_cf"]


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
plot_dual_axis(years, sentiment_lagged, errors_lagged, net_cf, "Absolute Change in Net Cashflow", f"Lagged Sentiment vs Absolute Change in Net Cashflow ({fund_name})", fund_number_name=fund_number_name)
plot_dual_axis(years, sentiment_lagged, errors_lagged, dpi, "DPI", f"Lagged Sentiment vs DPI ({fund_name})", fund_number_name=fund_number_name)
