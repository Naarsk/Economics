from Finance.Thesis.B_analysis_functions import load_sentiment, load_gp_cashflows
from Finance.Thesis.C_plotting_functions import plot_dual_axis

start_year, end_year = 2010, 2025

fund_name = "Apollo Global Management"
fund_number_name = "23_AGM"
freq = "yearly"

gp_path = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\GP_cashflows.xlsx"
irr_path = r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\Excels\Horizon_IRR.xlsx"
sentiment_path = rf"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel\{fund_number_name}.xlsx"
snp_path = r"C:\Users\leocr\Projects\Economics\Risk Management\data_sp500.xlsx"

sentiment_df = load_sentiment(sentiment_path, start_year, end_year, freq=freq, lag=2)
cashflow_df = load_gp_cashflows(gp_path, fund=fund_name, start_year= start_year, end_year= end_year,freq=freq)

# -----------------------------
# Merge everything into a single DataFrame
# -----------------------------
df = (
    sentiment_df[["sentiment_lagged", "errors_lagged"]]
    .merge(cashflow_df, on="period", how="inner")
)

print(df)

# -----------------------------
# Optional: filter to final start-end range
# -----------------------------
df = df.loc[(df.index >= start_year) & (df.index <= end_year)]
years=df.index
sentiment_lagged = df["sentiment_lagged"]
errors_lagged = df["errors_lagged"]
net_cf = df["net_cf"]
amount = df["amount"]
rel_change_net_cf = df["rel_change_net_cf"]

#amount 0.03
#rel_change_amount 0.073

# plots
plot_dual_axis(years, sentiment_lagged, errors_lagged, net_cf, "Net Cashflow", f"Lagged Sentiment vs Net Cashflow ({fund_name})", fund_number_name=fund_number_name)
# plot_dual_axis(years, sentiment_lagged, errors_lagged, dpi, "DPI", f"Lagged Sentiment vs DPI ({fund_name})", fund_number_name=fund_number_name)
# plot_dual_axis(years, sentiment_lagged, errors_lagged, abs_change_net_cf, "Absolute Change in Net Cashflow", f"Lagged Sentiment vs Absolute Change in Net Cashflow ({fund_name})", fund_number_name=fund_number_name)
plot_dual_axis(years, sentiment_lagged, errors_lagged, rel_change_net_cf, "Relative Change in Net Cashflow", f"Lagged Sentiment vs Relative Change in Net Cashflow ({fund_name})", fund_number_name=fund_number_name)
plot_dual_axis(years, sentiment_lagged, errors_lagged, amount, "Distributions", f"Lagged Sentiment vs Distributions ({fund_name})", fund_number_name=fund_number_name)
