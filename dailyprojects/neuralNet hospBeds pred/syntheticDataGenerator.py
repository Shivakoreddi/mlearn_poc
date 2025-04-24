##Patient wait time estimator

##Features - {Department, urgency_level, queue_length, Doctor_availability, patient_arrival_hour, Expected_wait_time}

import numpy as np
import pandas as pd

np.random.seed(42)
n = 50000

departments = ["Emergency","ICU","Surgery"]
day_of_weeks = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
current_occupancy = np.random.randint(1,20,n)
discharge_3days = np.random.randint(1,5,n)
admission_3days = np.random.randint(1,10,n)
holiday_flag = np.random.choice([0,1],size=n)
flu_season = np.random.choice([0,1],size=n)
department = np.random.choice(departments,n)
day_of_week = np.random.choice(day_of_weeks,n)



# wait_time = (
#     (50 - urgency * 10) +
#     queue * 0.8 +
#     (1 - doctor_avail) * 15 +
#     np.where((arrival_hour >= 9) & (arrival_hour <= 12), 10, 0) +
#     np.where((arrival_hour >= 17) & (arrival_hour <= 20), 10, 0) +
#     np.random.normal(0, 5, n)  # noise
# )
# wait_time = np.clip(wait_time, 0, None)  # No negative wait

##Simulated target: beds_needed_7days
beds_needed = (
    current_occupancy +
    np.where((day_of_week == 5) | (day_of_week == 6), admission_3days * 1.2 - discharge_3days * 0.8, admission_3days * 0.8 - discharge_3days * 1.1) +
    np.where((department == 0) & ((flu_season == 1) | (holiday_flag == 1)), 10, 0) +
    np.where((department == 1) | (department == 2), 5, 0) +
    np.random.normal(0, 5, n)
)
beds_needed = np.clip(beds_needed, 0, None)  # No negative wait


df = pd.DataFrame({
    "Department":department,
    "day":day_of_week,
    "occupancy":current_occupancy,
    "discharges":discharge_3days,
    "admissions":admission_3days,
    "holiday":holiday_flag,
    "flu":flu_season,
    "beds_needed":beds_needed.round(0)
})


print(df.head(100))

# Write to CSV
df.to_csv("synthetic_hospital_bed_forcast.csv", index=False)
