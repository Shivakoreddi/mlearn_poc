import numpy as np

import pandas as pd

data = pd.read_csv('hospital_data_raw.csv')
print(data.columns)
print(data.head(5))
print(data['department'].unique())

##group by with depratment for total _length of stay
print(data.groupby(['department','diagnosis_code'])['length_of_stay'].sum())

##transform datetime to data value and add as new column
##df = pd.DataFrame({'timestamp': ['2024-02-03 16:00:00']})
data['admission_datetime'] = pd.to_datetime(data['admission_datetime'])  # Ensure it's a datetime

# Convert to just the date
data['admission_date'] = data['admission_datetime'].dt.date

print(data.head(5))

##group by with department for total _length of stay
print(data.groupby(['department','admission_date','age'])['length_of_stay'].sum())


data['admission_date'] = pd.to_datetime(data['admission_date'], errors='coerce')
data = pd.get_dummies(data, columns=['department', 'diagnosis_code'], drop_first=True)


data['admission_day'] = data['admission_date'].dt.day_name()
data['admission_month'] = data['admission_date'].dt.month
data['admission_year'] = data['admission_date'].dt.year
data['is_weekend'] = data['admission_date'].dt.weekday >= 5  # Saturday/Sunday



#
# ##hospital load
# data['net_occupancy'] = data['current_occupancy'] + data['admissions_last_3_days'] - data['discharges_last_3_days']
#
# data['holiday_flu_risk'] = data['holiday'].astype(int) + data['flu_season'].astype(int)

print(data.info)

##group by with department for total _length of stay
print(data.groupby(['department','age'])['length_of_stay'].sum())

