import pandas as pd

from Finance.Thesis.functions import plot_avg_outlook_by_year, plot_distributions, get_most_used_words, \
    get_least_used_words

excel_path = r"C:\Users\leocr\Projects\Economics\Finance/Thesis/files/responses\excel\21_TPG.xlsx"
df = pd.read_excel(excel_path)
plot_avg_outlook_by_year(df, "21_TPG.png", 2010,2025)
#plot_distributions(df)

# word_df=get_most_used_words(df)
# word_df=get_least_used_words(df)

#print(word_df)