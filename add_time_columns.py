#!/usr/bin/env python3
"""
Script to add Trip_Departure_Time and Trip_Arrival_Time columns 
from the original Jaddah_Trips.csv to the optimized Trips.csv
"""

import pandas as pd

# Read the original trips file with time columns
print("Reading original trips data...")
original_trips = pd.read_csv('../csv_data/Jaddah_Trips.csv', usecols=['Trip_id', 'Trip_Departure_Time', 'Trip_Arrival_Time'])

# Read the optimized trips file
print("Reading optimized trips data...")
optimized_trips = pd.read_csv('csv_data_optimized/Trips.csv')

print(f"Original trips: {len(original_trips)} rows")
print(f"Optimized trips: {len(optimized_trips)} rows")

# Merge the time columns based on Trip_id
print("Merging time columns...")
merged_trips = optimized_trips.merge(
    original_trips[['Trip_id', 'Trip_Departure_Time', 'Trip_Arrival_Time']], 
    on='Trip_id', 
    how='left'
)

print(f"Merged trips: {len(merged_trips)} rows")
print(f"Trips with time data: {merged_trips['Trip_Departure_Time'].notna().sum()}")

# Reorder columns to put time columns after coordinates
column_order = [
    'Trip_id',
    'Person_id',
    'Family_Number',
    'Trip_Number',
    'Origin_Latitude',
    'Origin_Longitude',
    'Destination_Latitude',
    'Destination_Longitude',
    'Trip_Departure_Time',
    'Trip_Arrival_Time',
    'Trip_Origin_Purpose',
    'Trip_Destination_Purpose',
    'Mode_of_Travel_1'
]

merged_trips = merged_trips[column_order]

# Save the updated file
print("Saving updated trips data...")
merged_trips.to_csv('csv_data_optimized/Trips.csv', index=False)

print("Done! Time columns added successfully.")
print("\nSample of updated data:")
print(merged_trips[['Trip_id', 'Trip_Departure_Time', 'Trip_Arrival_Time', 'Trip_Origin_Purpose', 'Trip_Destination_Purpose']].head(10))
