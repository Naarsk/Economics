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
dta.insert(0,'Date_reported', italy_df['Date_reported'] )
dta.insert(2, 'Average_temperature',italy_df['Temperatura_Media'] )
dta.insert(3, 'Retail_and_recreation',italy_df['retail_and_recreation_percent_change_from_baseline'] )
dta.insert(4, 'Transit_stations',italy_df['transit_stations_percent_change_from_baseline'] )
dta.insert(5, 'Cumulative_moving_average', italy_df['Cumulative_cases'].shift(1).fillna(0).expanding().mean())
dta.insert(6, 'Parks',italy_df['parks_percent_change_from_baseline'] )
dta.insert(7,'New_cases', italy_df['New_cases'] )
dta = dta[dta['New_cases'] > 0]
dta.reset_index(drop=True, inplace=True)


# Export the DataFrame to a Stata .dta file.
dta.to_stata('../data/italy_data.dta', write_index=False)
