import pandas as pd

from Finance.Thesis.data_analysis import plot_distributions, plot_avg_outlook_by_year

excel_path = r"C:\Users\leocr\Projects\Economics\Finance\Thesis\files\responses\excel\clean_outlooks.xlsx"
df = pd.read_excel(excel_path)
plot_avg_outlook_by_year(df)
plot_distributions(df)