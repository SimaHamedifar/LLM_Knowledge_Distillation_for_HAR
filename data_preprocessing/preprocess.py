"""
This code preprocesses the MuRAL data and prepares it for training. 
"""

import glob
import os
import pandas as pd
import json

all_windows = []

for folder in sorted(glob.glob("MuRAL/*/")):
    csv_file = os.path.join(folder, "data.csv")
    df = pd.read_csv(csv_file)
    events = []
    for _, row in df.iterrows():
        sensor = str(row['sensor'])
        action = str(row['action'])

        events.append(
            {
                "event_id": float(row['uid']),
                "time": str(row['time']),
                "description": f"{sensor} {action}"
            }
        )

    for i in range(0, len(events), 10):
        window = events[i:i+10]
        if len(window) == 10:
            all_windows.append(window)

with open("preprocess_data.json", "w") as f:
    json.dump(all_windows, f, indent=4)