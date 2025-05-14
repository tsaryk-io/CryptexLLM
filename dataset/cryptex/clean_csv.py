import pandas as pd
import numpy as np

# Path to the CSV file
file_path = 'candlesticks-h.csv'

# Read the CSV file
df = pd.read_csv(file_path)

# Remove any rows with non-numeric data (including NaN)
df_cleaned = df.dropna()  # Remove NaNs first
df_cleaned = df_cleaned[df_cleaned.applymap(np.isreal).all(axis=1)]  # Keep only rows with all numeric data

# Save the cleaned DataFrame back to the original file
df_cleaned.to_csv(file_path, index=False)

print("CSV file has been cleaned and updated.")
