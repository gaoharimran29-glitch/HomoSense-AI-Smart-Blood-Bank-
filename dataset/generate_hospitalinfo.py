import pandas as pd
import numpy as np
import os

blood_groups = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]
nearest_hospitals = ["HOSP_B","HOSP_C","HOSP_D","HOSP_E" , "HOSP_F" , "HOSP_G" , "HOSP_H"]

# Central Hospital Coordinates (Your Location)
# We will generate nearby coordinates relative to this center
CENTER_LAT = 28.6139 
CENTER_LON = 77.2090

data = []
for bg in blood_groups:
    for hosp in nearest_hospitals:
        stock = np.random.randint(20, 100)
        distance = round(np.random.uniform(2, 15), 1)  # km
        
        # Supply Chain Data: Generating realistic nearby Lat/Lon
        lat_offset = np.random.uniform(-0.05, 0.05)
        lon_offset = np.random.uniform(-0.05, 0.05)
        lat = CENTER_LAT + lat_offset
        lon = CENTER_LON + lon_offset
        
        # Email for Request System
        email = "gaoharimran6202@gmail.com"
        
        data.append([hosp, bg, stock, distance, lat, lon, email])

df = pd.DataFrame(data, columns=[
    "HospitalID", "BloodGroup", "UnitsAvailable", 
    "Distance_km", "Lat", "Lon", "Email"
])

os.makedirs("dataset", exist_ok=True)
df.to_csv(r"dataset/nearest_hospitals.csv", index=False)
print("Nearest hospitals CSV updated with Supply Chain and Email data.")
print(df.head())