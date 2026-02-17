import pandas as pd
import numpy as np
import time
from datetime import datetime

BLOOD_GROUPS = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]
HOSPITAL = "HOSP_A"
TARGET_ROWS = 50  # for demo

while True:
    all_rows = []
    current_date = pd.Timestamp.now().normalize()  # only date part
    
    for blood in BLOOD_GROUPS:
        # Randomly pick stock scenario to generate diverse statuses
        scenario = np.random.choice(["safe","shortage","expiry","both"])
        
        if scenario == "safe":
            current_stock = np.random.randint(80,150)
            expired = np.random.randint(0,2)
        elif scenario == "shortage":
            current_stock = np.random.randint(5,30)
            expired = 0
        elif scenario == "expiry":
            current_stock = np.random.randint(50,100)
            expired = np.random.randint(2,10)
        else:  # both
            current_stock = np.random.randint(5,40)
            expired = np.random.randint(1,5)
        
        for i in range(TARGET_ROWS // len(BLOOD_GROUPS)):
            date = current_date - pd.Timedelta(minutes=i*5)
            weekend = 1 if date.weekday() >= 5 else 0
            holiday = 1 if date.month in [1,8,10] and date.day < 5 else 0
            temp = 25 + 10 * np.random.normal(0,1)
            traffic = np.random.uniform(0.3,0.9)
            emergency = int(traffic * np.random.randint(1,5))
            collection = np.random.randint(5,20)
            
            used = np.random.randint(5, min(current_stock, 15)) if current_stock>5 else current_stock
            units_available = max(current_stock + collection - expired - used, 0)
            current_stock = units_available  # for next iteration
            
            all_rows.append([
                date, HOSPITAL, blood,
                units_available, collection, used, expired,
                emergency, round(traffic,2), round(temp,2), holiday, weekend
            ])
    
    df_stock = pd.DataFrame(all_rows, columns=[
        "Date","HospitalID","BloodGroup","UnitsAvailable",
        "UnitsCollected","UnitsUsed","UnitsExpired",
        "EmergencyCases","TrafficIndex","AvgTemperature",
        "HolidayFlag","WeekendFlag"
    ])
    
    df_stock.to_csv(r"dataset/current_stock.csv", index=False)
    print(f"✅ Current stock updated at {current_date}")
    
    time.sleep(10)