from Finance.Thesis.C_plotting_functions import plot_avg_outlook

fund_number_name = "21_TPG"

excel_path = rf"C:\Users\leocr\Projects\Economics\Finance/Thesis/files/responses\excel\{fund_number_name}.xlsx"

plot_avg_outlook(excel_path, f"{fund_number_name}.png", 2017,2025, freq="yearly")
