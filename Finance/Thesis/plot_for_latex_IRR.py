import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Finance.Thesis.palette import palette


# --- Load S&P500 daily close ---
sp500_df = pd.read_excel(r"C:\Users\leocr\Projects\Economics\Risk Management\data_sp500.xlsx")
sp500_df["Date"] = pd.to_datetime(sp500_df["Date"])
sp500_df = sp500_df.set_index("Date")

# Annual log returns
sp500_annual = sp500_df["Close"].resample("YE").last()
sp500_annual_logret = np.log(sp500_annual / sp500_annual.shift(1))
sp500_annual_logret = sp500_annual_logret.loc["2000":]  # from Jan 2000 onwards

print(sp500_annual_logret)

# --- Load Private Equity IRR ---
pe_df = pd.read_excel(r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\Excels\Horizon_IRR.xlsx")
pe_df.iloc[:, 0] = pd.to_datetime(pe_df.iloc[:, 0])
pe_df = pe_df.set_index(pe_df.columns[0])
pe_1yr_irr = pe_df.iloc[:, 0] / 100  # convert % to decimal if needed
pe_1yr_irr.dropna(inplace=True)

print(pe_1yr_irr)

# --- Align on common years ---
combined = pd.concat([
    sp500_annual_logret.rename("SNP500_LogRet"),
    pe_1yr_irr.rename("PE_1yr_IRR")
], axis=1).dropna()

# --- Plot ---
plt.figure(figsize=(10,6))
plt.plot(combined.index.year, combined["SNP500_LogRet"], label="S&P 500 (Annual Log Return)", marker='o', color=palette["dark_gray"])
plt.plot(combined.index.year, combined["PE_1yr_IRR"], label="Private Equity (1Y IRR)", marker='o', color=palette["primary_red"])

# Crisis shading
plt.axvspan(2008, 2009, color=palette["soft_gray"], alpha=0.3, label="GFC")
plt.axvspan(2020, 2022, color="red", alpha=0.2, label="COVID")

plt.axhline(0, color="black", linewidth=1)
plt.title("Public vs. Private Equity Annual Returns", fontsize=14)
plt.ylabel("Annual Return")
plt.xlabel("Year")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()

plt.savefig(r"C:\Users\leocr\Projects\Economics\Finance\Thesis\Latex\img\pe_vs_pm.png")
plt.show()
