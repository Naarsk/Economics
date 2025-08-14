import pandas as pd
import matplotlib.pyplot as plt
import os


# Custom color palette
palette = {
    "primary_red": "#C00000",
    "dark_gray": "#4D4D4D",
    "soft_gray": "#A6A6A6",
    "accent_orange": "#E07B39",
    "deep_blue": "#003366",
    "muted_green": "#3A7D44"
}

# Example usage in plots:
# ax.bar(..., color=palette["primary_red"])
# ax.plot(..., color=palette["deep_blue"])

# File path
file_path = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\data\data_for_plots.xlsx"
output_dir = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\Latex\img"
os.makedirs(output_dir, exist_ok=True)  # ensure directory exists

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file was not found: {file_path}")

# Read Excel file
df = pd.read_excel(file_path)

# Clean YEAR column (remove ' YTD' and convert to int where possible)
df['YEAR'] = df['YEAR'].astype(str).str.replace(' YTD', '', regex=False)
df = df[df['YEAR'].str.isnumeric()]
df['YEAR'] = df['YEAR'].astype(int)

# Sort by year
df = df.sort_values('YEAR')

# === Plot 1: Total Fundraised vs Average Fundraising Size ===
fig, ax1 = plt.subplots(figsize=(12,6))

# Bar for total aggregate capital raised
ax1.bar(df['YEAR'], df['AGGREGATE CAPITAL RAISED (USD BN)'], color=palette["soft_gray"], label="Total Capital Raised (USD BN)", alpha=0.7)
ax1.set_xlabel("Year")
ax1.set_ylabel("Total Capital Raised (USD BN)")
ax1.tick_params(axis='y')

# Secondary axis for average fundraising size
ax2 = ax1.twinx()
ax2.plot(df['YEAR'], df['AVERAGE FUNDRAISING SIZE (USD MN)'], color=palette["primary_red"], marker="o", label="Average Fundraising Size (USD MN)")
ax2.set_ylabel("Average Fundraising Size (USD MN)")

# Legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.title("Total Fundraised and Average Fundraising Size by Year")
plt.tight_layout()

plot1_path = os.path.join(output_dir, "fundraising_vs_average.png")
plt.savefig(plot1_path, dpi=300, bbox_inches="tight")
plt.show()

# === Plot 2: AUM with Dry Powder and Unrealized Value ===
fig, ax = plt.subplots(figsize=(12,6))

# Stacked bars: dry powder + unrealized value
ax.bar(df['YEAR'], df['DRY POWDER'], color=palette["dark_gray"], label="Dry Powder", alpha=0.7)
ax.bar(df['YEAR'], df['UNREALIZED VALUE'], color=palette["soft_gray"], bottom=df['DRY POWDER'], label="Unrealized Value", alpha=0.7)

# Outline total AUM as a line
ax.plot(df['YEAR'], df['ASSETS UNDER MANAGEMENT'], color=palette["primary_red"], marker="o", linewidth=2, label="Total AUM")

ax.set_xlabel("Year")
ax.set_ylabel("USD BN")
plt.title("Assets Under Management with Dry Powder and Unrealized Value")
plt.legend()
plt.tight_layout()

plot2_path = os.path.join(output_dir, "aum_drypowder_unrealized.png")
plt.savefig(plot2_path, dpi=300, bbox_inches="tight")
plt.show()

