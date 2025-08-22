import statsmodels.api as sm

from Finance.Thesis.B_analysis_functions import load_sentiment, load_pe_irr, load_gp_cashflows, load_snp500

start_year, end_year = 2010, 2025

fund_name = "TPG"
fund_number_name = "21_TPG"
freq = "yearly"

gp_path = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\GP_cashflows.xlsx"
irr_path = r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\Excels\Horizon_IRR.xlsx"
sentiment_path = rf"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel\{fund_number_name}.xlsx"
snp_path = r"C:\Users\leocr\Projects\Economics\Risk Management\data_sp500.xlsx"

# pe_irr = load_pe_irr(irr_path,  start_year, end_year)
# snp500 = load_snp500(snp_path, start_year=start_year, end_year=end_year, freq = freq)
sentiment_df = load_sentiment(sentiment_path, start_year, end_year, freq=freq, lag=1)
cashflow_df = load_gp_cashflows(gp_path, fund=fund_name, start_year= start_year, end_year= end_year,freq=freq)

# -----------------------------
# Merge everything into a single DataFrame
# -----------------------------
df = (
    sentiment_df[["sentiment_lagged", "errors_lagged"]]
    #.merge(pe_irr, on="period", how="inner")
    #.merge(snp500, on="period", how="inner")
    .merge(cashflow_df, on="period", how="inner")
)

# -----------------------------
# Optional: filter to final start-end range
# -----------------------------
df = df.loc[(df.index >= start_year) & (df.index <= end_year)]

print("Merged DataFrame:")
print(df.head())

df.dropna(inplace=True)
print(df)


y = df["sentiment_lagged"]
X = df["amount"]

X = sm.add_constant(X)
ols_results = sm.OLS(y, X).fit()
print(ols_results.summary())


y = df["sentiment_lagged"]
X = df["rel_change_net_cf"]

X = sm.add_constant(X)
ols_results = sm.OLS(y, X).fit()
print(ols_results.summary())


y = df["sentiment_lagged"]
X = df["rel_change_amount"]

X = sm.add_constant(X)
ols_results = sm.OLS(y, X).fit()
print(ols_results.summary())