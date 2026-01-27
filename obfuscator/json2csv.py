import json
import pandas as pd

# Load the JSON data
#key = 'orignal_sample'
key = 'adversarial_sample'
file_path = 'result/adversarial_samples_.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# Prepare the list for DataFrame
rows = []
for item in data:
    row = {}
    row['orignal_index'] = item['orignal_index']
    row = item[key]
    row['orignal_label'] = item['orignal_label']
    rows.append(row)

# Create the DataFrame
df = pd.DataFrame(rows)

# Save to CSV
df.to_csv('result/adversarial_samples.csv', index=False)
#df.to_csv('result/orignal_samples.csv', index=False)
