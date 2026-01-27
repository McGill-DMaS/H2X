import pandas as pd
import glob
import os

# Base directory
base_dir = "E:\\saqib_work1\\data\\miles\\feature_extracted_csv\\"
#base_dir = "C:\\Users\\Saqib\\PycharmProjects\\H2X\\obfuscator\\result"

# List to hold all CSV file paths
all_files = []

# Walk through directory tree
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".csv"):
            all_files.append(os.path.join(root, file))

# Merge all CSV files
df = pd.concat(map(pd.read_csv, all_files), ignore_index=True)

# Output file path
#foutput = "C:\\Users\\Saqib\\PycharmProjects\\H2X\\obfuscator\\result\\merge_csv_samples.csv"
foutput = "C:\\Users\\Saqib\\PycharmProjects\\H2X\\model\\dataset\\merge_csv_samples_20240809.csv"
df.to_csv(foutput, index=False)

print("DONE!!")
