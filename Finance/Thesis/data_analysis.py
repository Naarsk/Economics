import matplotlib.pyplot as plt
import pandas as pd

def plot_avg_outlook_by_quarter(df: pd.DataFrame):
    """Calculates average outlook_num by quarter and plots it."""

    # ✅ Ensure date is datetime
    df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df[df["date_dt"].notna()]

    # ✅ Extract Year-Quarter
    df["quarter"] = df["date_dt"].dt.to_period("Q")

    # ✅ Calculate average per quarter
    avg_by_quarter = df.groupby("quarter")["outlook_num"].mean()

    # ✅ Plot
    plt.figure(figsize=(10, 5))
    avg_by_quarter.plot(marker="o")
    plt.title("Average Outlook by Quarter")
    plt.xlabel("Quarter")
    plt.ylabel("Average Outlook (1=Increase, 0=Stable, -1=Decrease)")
    plt.grid(True)
    plt.show()

def plot_avg_outlook_by_year(df: pd.DataFrame):
    """Calculates average outlook_num by year (2019–2025) and plots it."""

    # ✅ Ensure date is datetime
    df["date_dt"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
    df = df[df["date_dt"].notna()]

    # ✅ Extract year and filter range
    df["year"] = df["date_dt"].dt.year
    df = df[(df["year"] >= 2019) & (df["year"] <= 2025)]

    # ✅ Calculate average per year
    avg_by_year = df.groupby("year")["outlook_num"].mean()

    # ✅ Plot
    plt.figure(figsize=(8, 5))
    avg_by_year.plot(marker="o", color="blue")
    plt.title("Average Outlook by Year (2019–2025)")
    plt.xlabel("Year")
    plt.ylabel("Average Outlook (1=Increase, 0=Stable, -1=Decrease)")
    plt.grid(True)
    plt.show()


def plot_distributions(df: pd.DataFrame):
    """Plots histogram distributions for outlook_num and confidence."""

    plt.figure(figsize=(12, 5))

    # ✅ Outlook_num distribution
    plt.subplot(1, 2, 1)
    df["outlook_num"].hist(bins=3, rwidth=0.8)
    plt.xticks([-1, 0, 1], ["Decrease (-1)", "Stable (0)", "Increase (1)"])
    plt.title("Distribution of Outlook")
    plt.xlabel("Outlook")
    plt.ylabel("Count")

    # ✅ Confidence distribution
    plt.subplot(1, 2, 2)
    df["confidence"].hist(bins=20, rwidth=0.8)
    plt.title("Distribution of Confidence")
    plt.xlabel("Confidence")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.show()
