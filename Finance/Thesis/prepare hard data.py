import pandas as pd
import os

def filter_gp_cashflows():
    # Input and output paths
    input_path = r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\Data\Data_Preqin_Raw\Cashflows.xlsx"
    output_path = r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\GP_cashflows.xlsx"

    # List of GPs to filter for
    target_gps = [
        "TPG",
        "Carlyle Group",
        "Apollo Global Management",
        "Blackstone Group",
        "CVC",
        "Apax Partners",
        "Goldman Sachs Asset Management",
        "Bain Capital",
        "KKR",
        "Advent International",
        "Permira",
        "Silver Lake",
        "Providence Equity Partners",
        "Madison Dearborn Partners",
        "EQT",
        "Nordic Capital",
        "Welsh, Carson, Anderson & Stowe",
        "Leonard Green & Partners",
        "Charterhouse Capital Partners",
        "Onex"
    ]

    # Columns to retain
    columns_to_keep = [
        "FUND ID", "FIRM ID", "NAME", "ASSET CLASS", "FUND MANAGER", "REGION",
        "VINTAGE / INCEPTION YEAR", "STRATEGY", "FUND SIZE (USD MN)", "STATUS",
        "PRIMARY REGION FOCUS", "CORE INDUSTRIES", "TRANSACTION TYPE", "TRANSACTION DATE",
        "TRANSACTION AMOUNT", "CUMULATIVE CONTRIBUTION", "CUMULATIVE DISTRIBUTION",
        "NET CASHFLOW", "QUARTILE", "INDUSTRIES", "INDUSTRY VERTICALS",
        "GEOGRAPHIC EXPOSURE", "GEOGRAPHIC FOCUS", "OTHER GEOGRAPHIES",
        "FUND CURRENCY", "TARGET SIZE CURR (MN)", "FINAL CLOSE SIZE CURR (MN)"
    ]

    # Read Excel file
    df = pd.read_excel(input_path, engine='openpyxl')

    # Filter rows where FUND MANAGER matches one of the target GPs
    filtered_df = df[df['FUND MANAGER'].isin(target_gps)]

    # Keep only the desired columns
    filtered_df = filtered_df[columns_to_keep]

    # Export to new Excel file
    filtered_df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"Filtered data saved to: {output_path}")


def filter_lp_cashflows():
    # Input and output paths
    input_path = r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\Data\Data_Preqin_Raw\Cashflows.xlsx"
    output_path = r"D:\Files\OneDrive - University of Luxembourg\Thesis\PreqinDownloads\LP_cashflows.xlsx"

    # List of LPs to filter
    target_lps = [
        "California Public Employees' Retirement System",
        "CPP Investments",
        "California State Teachers' Retirement System",
        "Washington State Investment Board",
        "Oregon Public Employees Retirement Fund",
        "New York State Common Retirement Fund",
        "Teacher Retirement System of Texas",
        "Florida State Board of Administration",
        "Hamilton Lane",
        "Pennsylvania Public School Employees' Retirement System",
        "Michigan Department of Treasury",
        "State of Wisconsin Investment Board",
        "Massachusetts Pension Reserves Investment Management Board",
        "Teachers Retirement System of Georgia",
        "Teachers' Retirement System of the State of Illinois",
        "State Teachers Retirement System of Ohio",
        "Los Angeles County Employees' Retirement Association",
        "New York State Teachers' Retirement System",
        "Maryland State Retirement and Pension System",
        "Minnesota State Board of Investment"
    ]

    # Columns to retain
    columns_to_keep = [
        "FUND ID", "FIRM ID", "NAME", "ASSET CLASS", "FUND MANAGER", "REGION",
        "VINTAGE / INCEPTION YEAR", "STRATEGY", "FUND SIZE (USD MN)", "STATUS",
        "PRIMARY REGION FOCUS", "CORE INDUSTRIES", "TRANSACTION TYPE", "TRANSACTION DATE",
        "TRANSACTION AMOUNT", "CUMULATIVE CONTRIBUTION", "CUMULATIVE DISTRIBUTION",
        "NET CASHFLOW", "QUARTILE", "INDUSTRIES", "INDUSTRY VERTICALS",
        "GEOGRAPHIC EXPOSURE", "GEOGRAPHIC FOCUS", "OTHER GEOGRAPHIES",
        "FUND CURRENCY", "TARGET SIZE CURR (MN)", "FINAL CLOSE SIZE CURR (MN)"
    ]

    # Load data
    df = pd.read_excel(input_path, engine='openpyxl')

    # --- Determine LP column name ---
    # You can uncomment below to inspect columns:
    # print(df.columns)

    # Adjust this if your column is named differently
    lp_column = "FUND MANAGER"

    # Filter LPs
    filtered_df = df[df[lp_column].isin(target_lps)]

    # Keep only selected columns (only those that exist in df)
    filtered_df = filtered_df[[col for col in columns_to_keep if col in df.columns]]

    # Save to Excel
    filtered_df.to_excel(output_path, index=False, engine='openpyxl')

    print(f"Filtered LP data saved to: {output_path}")

# Run it
filter_gp_cashflows()

