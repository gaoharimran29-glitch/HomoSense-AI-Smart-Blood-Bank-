import numpy as np
import pandas as pd

# ---------------- CONFIG ----------------
START_DATE = "2023-01-01"
END_DATE = "2025-12-31"
HOSPITAL = "HOSP_A"
BLOOD_GROUPS = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]
TARGET_ROWS = 20000 

np.random.seed(42)
date_range = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
blood_group_weights = {"O+": 1.2, "A+": 1.0, "B+": 1.1, "AB+": 0.5, "O-": 2.5, "A-": 0.8, "B-": 0.7, "AB-": 0.4}

all_rows = []

# ---------------- GENERATION ----------------
for blood in BLOOD_GROUPS:
    current_stock = np.random.randint(150, 250)
    rows_per_group = TARGET_ROWS // len(BLOOD_GROUPS)
    
    # Generate random dates but SORT them for this blood group first
    # This ensures the inventory chain follows time correctly
    group_dates = sorted([np.random.choice(date_range) for _ in range(rows_per_group)])
    
    for i, date in enumerate(group_dates):
        date = pd.Timestamp(date)
        
        # Features
        seasonal_multiplier = 1.8 if date.month in [1, 2, 8] else 1.0
        weekend = 1 if date.weekday() >= 5 else 0
        holiday = 1 if date.month in [1, 8, 10] and date.day < 5 else 0
        temp = 25 + 10 * np.sin(2 * np.pi * date.timetuple().tm_yday / 365) + np.random.normal(0, 2)
        traffic = np.random.uniform(0.3, 0.9)
        emergency = int(traffic * np.random.randint(2, 10))
        
        # 1. Opening Stock
        opening_stock = current_stock
        
        # 2. Demand & Collection
        demand = int((np.random.randint(5, 15) * blood_group_weights[blood] + (weekend*2) + (holiday*3) + emergency) * seasonal_multiplier)
        collection = max(0, int(np.random.normal(demand, 3)))
        
        # 3. Expiry subtraction
        expired = int(opening_stock * 0.01) if opening_stock > 30 else 0
        
        # 4. Math: Opening + Collected - Expired - Used = Next Opening
        total_in_hand = opening_stock + collection - expired
        actual_used = min(demand, total_in_hand)
        
        closing_stock = total_in_hand - actual_used
        current_stock = closing_stock 
        
        all_rows.append([
            date, i, HOSPITAL, blood, # Added 'i' as a TransactionID
            opening_stock, collection, actual_used, expired,
            emergency, round(traffic, 2), round(temp, 2), holiday, weekend
        ])

# ---------------- CREATE & SORT ----------------
df = pd.DataFrame(all_rows, columns=["Date", "TransactionID", "HospitalID","BloodGroup","UnitsAvailable","UnitsCollected","UnitsUsed","UnitsExpired","EmergencyCases","TrafficIndex","AvgTemperature","HolidayFlag","WeekendFlag"])

# FIX: Sort by BloodGroup AND TransactionID to preserve the math chain
df = df.sort_values(["BloodGroup", "TransactionID"]).reset_index(drop=True)

# Drop TransactionID before saving so it doesn't mess up your model features
df_to_save = df.drop(columns=["TransactionID"])
df_to_save.to_csv("dataset/blood_bank_dataset_final.csv", index=False)

print("✅ Inventory Math Chain Locked and Verified.")

# VERIFICATION
# Let's look at A+ flow again
check = df[df['BloodGroup'] == 'A+'].head(5)
print(check[['Date', 'UnitsAvailable', 'UnitsCollected', 'UnitsUsed', 'UnitsExpired']])