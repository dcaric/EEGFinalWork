#initial load data file

import pandas as pd
import os

file_path = 'DatasetSub_shorten.xlsx'

if os.path.exists(file_path):
    print(f"Loading {file_path}...")
    try:
        df = pd.read_excel(file_path)
        print("Data loaded successfully!")
        print("-" * 30)
        print(df.head())
        print("-" * 30)
        print(df.info())
    except Exception as e:
        print(f"Error loading file: {e}")
else:
    print(f"File not found: {file_path}")
