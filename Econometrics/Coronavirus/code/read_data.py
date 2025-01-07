import pandas as pd

df= pd.read_csv('../WHO-COVID-19-global-daily-data.csv')
start_date = '2020-02-15'
end_date = '2020-07-15'
italy_df = df[(df['Country'] == 'Italy') &
              (df['Date_reported'] >= start_date) &
              (df['Date_reported'] <= end_date)]
dta=pd.DataFrame(italy_df['New_cases'].fillna(0))
dta.insert(1, 'New_cases_lag_1', dta['New_cases'].shift(1).fillna(0))
dta.insert(2, 'New_cases_lag_2', dta['New_cases'].shift(2).fillna(0))
dta.insert(3, 'New_cases_lag_3', dta['New_cases'].shift(3).fillna(0))
