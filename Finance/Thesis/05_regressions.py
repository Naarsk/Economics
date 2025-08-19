from Finance.Thesis.B_analysis_functions import load_sentiment, load_pe_irr, load_gp_cashflows
import statsmodels.api as sm
import pandas as pd

fund_name = "Carlyle Group"
fund_number_name = "22_TCG"

def weighted_regression(y, X, weights):
    # Ensure X is a DataFrame so multiple regressors work
    if isinstance(X, pd.Series):
        X = X.to_frame()

    # Drop NaNs consistently across y, X, and weights
    mask = (~y.isna()) & (~weights.isna())
    for col in X.columns:
        mask &= ~X[col].isna()

    y, X, w = y[mask], X[mask], weights[mask]

    # Add constant for intercept
    X = sm.add_constant(X)

    model = sm.WLS(y, X, weights=w)
    results = model.fit()
    return results

gp_path = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\GP_cashflows.xlsx"

start_year, end_year = 2015, 2025
years = range(start_year, end_year + 1)


sentiment_df = load_sentiment(
    rf"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel\{fund_number_name}.xlsx",
    start_year, end_year, freq="yearly"
)
sentiment =sentiment_df["mean"]
weights = sentiment_df["counts"]

pe_irr = load_pe_irr(
    r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\Excels\Horizon_IRR.xlsx",
    years
)

net_cf, dpi = load_gp_cashflows(gp_path, fund=fund_name)

df = pd.concat(
    [
        sentiment.rename("sentiment"),
        pe_irr.rename("pe_irr"),
        net_cf.rename("net_cf"),
        dpi.rename("dpi"),
        weights
    ],
    axis=1
)

df["net_cf_lag1"] = df["net_cf"].shift(1)
df["dpi_lag1"]    = df["dpi"].shift(1)

df.dropna(inplace=True)

X = df[["net_cf_lag1", "dpi_lag1"]]
y = df["sentiment"]


X = sm.add_constant(X)
ols_results = sm.OLS(y, X).fit()
print(ols_results.summary())
