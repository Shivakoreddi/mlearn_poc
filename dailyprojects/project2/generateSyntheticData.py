##Patient wait time estimator

##Features - {Department, urgency_level, queue_length, Doctor_availability, patient_arrival_hour, Expected_wait_time}

import numpy as np
import pandas as pd

np.random.seed(42)
n = 50000

departments = ["Cardiology","Radiology","Pediatrics","General"]
urgency = np.random.randint(1,4,n)
queue = np.random.randint(0,51,n)
doctor_avail = np.random.choice([0,1],size=n)
arrival_hour = np.random.randint(0,24,n)
department = np.random.choice(departments,n)



# Simulated target: wait_time
# Heuristic:
# - High urgency = shorter wait
# - Longer queue = more wait
# - No doctor = more wait
# - Busy hours (9am–12pm, 5–8pm) = longer wait
wait_time = (
    (50 - urgency * 10) +
    queue * 0.8 +
    (1 - doctor_avail) * 15 +
    np.where((arrival_hour >= 9) & (arrival_hour <= 12), 10, 0) +
    np.where((arrival_hour >= 17) & (arrival_hour <= 20), 10, 0) +
    np.random.normal(0, 5, n)  # noise
)
wait_time = np.clip(wait_time, 0, None)  # No negative wait

df = pd.DataFrame({
    "Department":department,
    "Urgency_level":urgency,
    "Queue_length":queue,
    "Doctor_Availability":doctor_avail,
    "Patient_Arrival_hour":arrival_hour,
    "Expected_wait_time":wait_time.round(2)
})


print(df.head(100))



# Write to CSV
df.to_csv("synthetic_patient_wait_time.csv", index=False)
