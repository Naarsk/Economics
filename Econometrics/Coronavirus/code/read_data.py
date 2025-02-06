import numpy as np
import pandas as pd
import os

# Get the current directory
current_dir = os.path.dirname(__file__)

# Set the current directory as the working directory
os.chdir(current_dir)

# New Cases Dataset
df = pd.read_csv('../data/WHO-COVID-19-global-daily-data.csv')
start_date = '2020-02-15'
end_date = '2020-07-15'
italy_df = df[(df['Country'] == 'Italy') &
              (df['Date_reported'] >= start_date) &
              (df['Date_reported'] <= end_date)]

# Temperature Dataset
df_temp = pd.read_csv('../data/italy_data_temperature.csv')
df_temp.loc[:, 'time'] = pd.to_datetime(df_temp['time'])
italy_df.loc[:, 'Date_reported'] = pd.to_datetime(italy_df['Date_reported'])

# Merge the two DataFrames on the common date
italy_df = italy_df.merge(df_temp, left_on='Date_reported', right_on='time', how='inner')


# Mobility Dataset

df_mob = pd.read_csv('../data/italy_data_mobility.csv')
df_mob = df_mob[['date', 'retail_and_recreation_percent_change_from_baseline', 'grocery_and_pharmacy_percent_change_from_baseline', 'parks_percent_change_from_baseline', 'transit_stations_percent_change_from_baseline', 'workplaces_percent_change_from_baseline', 'residential_percent_change_from_baseline']]
df_mob.loc[:, 'date'] = pd.to_datetime(df_mob['date'])

# Group by 'date' and calculate the mean for each group
df_mob = df_mob.groupby('date').mean().reset_index()

italy_df['Date_reported'] = pd.to_datetime(italy_df['Date_reported'])

# Merge the two DataFrames on the common date
merged_df = italy_df.merge(df_mob, left_on='Date_reported', right_on='date', how='inner')

# Drop duplicate date columns (keeping one of them)
italy_df = merged_df.drop(columns=['time', 'date'])


dta = pd.DataFrame(italy_df['Cumulative_cases'].fillna(0))
dta.insert(1, 'Cumulative_cases_lag_1', dta['Cumulative_cases'].shift(1).fillna(0))
dta.insert(2, 'Cumulative_cases_lag_2', dta['Cumulative_cases'].shift(2).fillna(0))
dta.insert(3, 'Cumulative_cases_lag_3', dta['Cumulative_cases'].shift(3).fillna(0))
dta.insert(4, 'Cumulative_cases_lag_4', dta['Cumulative_cases'].shift(4).fillna(0))
dta.insert(5, 'Cumulative_cases_lag_5', dta['Cumulative_cases'].shift(5).fillna(0))
dta.insert(6, 'Cumulative_cases_lag_10', dta['Cumulative_cases'].shift(10).fillna(0))
dta.insert(7, 'Cumulative_cases_lag_20', dta['Cumulative_cases'].shift(20).fillna(0))
dta.insert(8, 'Cumulative_cases_lag_1_logX', np.log(dta['Cumulative_cases'].shift(1).fillna(0) + 10 ** (-6)))
dta.insert(9, 'Cumulative_cases_lag_1_XlogX',
           dta['Cumulative_cases'].shift(1).fillna(0) * np.log(dta['Cumulative_cases'].shift(1).fillna(0) + 10 ** (-6)))
dta.insert(10, 'Cumulative_cases_lag_1_Hermite2', dta['Cumulative_cases'].shift(1).fillna(0) ** 2 - np.average(
    dta['Cumulative_cases'].shift(1).fillna(0) ** 2))
dta.insert(11, 'Cumulative_cases_lag_1_Hermite3', dta['Cumulative_cases'].shift(1).fillna(0) ** 3 - 3 * np.average(
    dta['Cumulative_cases'].shift(1).fillna(0) ** 2))
dta.insert(12, 'Average_temperature',italy_df['Temperatura_Media'] )
dta.insert(13, 'Retail_and_recreation',italy_df['retail_and_recreation_percent_change_from_baseline'] )
dta.insert(14, 'Transit_stations',italy_df['transit_stations_percent_change_from_baseline'] )
dta.insert(15, 'Cumulative_cases_lag_10_XlogX',
           dta['Cumulative_cases'].shift(10).fillna(0) * np.log(dta['Cumulative_cases'].shift(10).fillna(0) + 10 ** (-6)))
dta.insert(15, 'Cumulative_moving_average', italy_df['Cumulative_cases'].shift(1).fillna(0).expanding().mean())
dta.insert(16, 'Cumulative_moving_sum', italy_df['Cumulative_cases'].shift(1).fillna(0).expanding().mean())
dta.insert(17, 'New_cases_lag_1', italy_df['New_cases'].shift(1).fillna(0))


# Export the DataFrame to a Stata .dta file.
dta.to_stata('../data/italy_data.dta', write_index=False)
