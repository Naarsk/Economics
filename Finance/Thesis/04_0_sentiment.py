import pandas as pd

from Finance.Thesis.C_plotting_functions import plot_avg_outlook_by_year

fund_number_name = "22_TCG"

excel_path = rf"C:\Users\leocr\Projects\Economics\Finance/Thesis/files/responses\excel\{fund_number_name}.xlsx"
df = pd.read_excel(excel_path)
plot_avg_outlook_by_year(df, f"{fund_number_name}.png", 2017,2025)
