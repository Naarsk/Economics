from linearmodels.panel import PanelOLS
import pandas as pd
import statsmodels.api as sm

# --- Prepare your dataset ---
df = pd.read_csv("panel_dataset.csv")

# Convert to MultiIndex for panel data
df["Quarter"] = pd.to_datetime(df["Quarter"])
df = df.set_index(["FUND ID", "Quarter"])
df = df.sort_index()

# Optionally, lag regressors by 1 quarter
df["Sentiment_lag1"] = df.groupby(level=0)["Sentiment"].shift(1)
df["Rm_lag1"] = df.groupby(level=0)["Rm"].shift(1)
df["Rf_lag1"] = df.groupby(level=0)["Rf"].shift(1)

# Drop missing due to lagging
df = df.dropna(subset=["NetFlowRatio", "Sentiment_lag1", "Rm_lag1", "Rf_lag1"])

# --- Regression: NetFlowRatio ~ Sentiment + Controls + FE ---
exog = sm.add_constant(df[["Sentiment_lag1", "Rm_lag1", "Rf_lag1"]])
model = PanelOLS(df["NetFlowRatio"], exog, entity_effects=True)
results = model.fit(cov_type="clustered", cluster_entity=True)

print(results.summary)
