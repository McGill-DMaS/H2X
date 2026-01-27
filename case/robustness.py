import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load SHAP explanations from JSON file
with open('result/H2X_explanations_obfuscator.json', 'r') as f:
    shap_explanations = json.load(f)

# Function to convert nested SHAP explanations into flat dictionaries
def convert_to_flat_dict(data):
    flat_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                for i, subval in enumerate(subvalue):
                    flat_data[f"{key}_{subkey}_{i}"] = subval
        else:
            flat_data[key] = value
    return flat_data

# Compute MMD score between benign and malware samples
def compute_mmd(x, y, sigma):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of data.

    Parameters:
    - x, y: arrays of shape (n_samples, n_features) representing two sets of data.
    - sigma: float, the bandwidth parameter of the RBF kernel.

    Returns:
    - float, the MMD between the two sets of data.
    """
    kxx = rbf_kernel(x, x, gamma=1.0 / (2.0 * sigma ** 2))
    kxy = rbf_kernel(x, y, gamma=1.0 / (2.0 * sigma ** 2))
    kyy = rbf_kernel(y, y, gamma=1.0 / (2.0 * sigma ** 2))
    mmd = np.mean(kxx) - 2.0 * np.mean(kxy) + np.mean(kyy)
    return mmd

# Calculate MMD for different values of n
def calculate_mmd_for_n_values(n_values, sigma=1.0):
    mmd_scores = []

    for n in n_values:
        shap_benign = shap_explanations['result_benign'][:n]
        shap_malware = shap_explanations['result_malware'][:n]

        df_benign_n = [convert_to_flat_dict(sample['data']) for sample in shap_benign]
        df_malware_n = [convert_to_flat_dict(sample['data']) for sample in shap_malware]

        # Convert the dictionaries into lists of lists
        df_benign_n_com_list = [list(sample.values()) for sample in df_benign_n]
        df_malware_n_com_list = [list(sample.values()) for sample in df_malware_n]

        # Flatten the lists if they contain nested lists
        df_benign_n_com_list = [np.concatenate(sample).tolist() for sample in df_benign_n_com_list]
        df_malware_n_com_list = [np.concatenate(sample).tolist() for sample in df_malware_n_com_list]

        # Convert to NumPy arrays and replace infinite values with NaN
        df_benign_n_com_list = np.array(df_benign_n_com_list)
        df_malware_n_com_list = np.array(df_malware_n_com_list)

        df_benign_n_com_list[np.isinf(df_benign_n_com_list)] = np.nan
        df_malware_n_com_list[np.isinf(df_malware_n_com_list)] = np.nan

        # Drop NaN values
        df_benign_n_com_list = pd.DataFrame(df_benign_n_com_list).dropna(axis=1).values
        df_malware_n_com_list = pd.DataFrame(df_malware_n_com_list).dropna(axis=1).values

        # Normalize the features
        scaler = StandardScaler()
        all_data = np.vstack((df_benign_n_com_list, df_malware_n_com_list))
        all_data = scaler.fit_transform(all_data)
        df_benign_n_com_list = all_data[:len(df_benign_n_com_list)]
        df_malware_n_com_list = all_data[len(df_benign_n_com_list):]

        mmd = compute_mmd(df_benign_n_com_list, df_malware_n_com_list, sigma)
        mmd_scores.append((n, mmd))
        print(f'MMD for n={n}: {mmd}')

    return mmd_scores

if __name__ == "__main__":
    n_values = list(range(10, 1010, 10))
    sigma = 1.0

    mmd_scores = calculate_mmd_for_n_values(n_values, sigma)

    # Save the results in a CSV file
    df = pd.DataFrame(mmd_scores, columns=['n', 'Robustness score'])
    df.to_csv('result/consistency_scores_obfuscator.csv', index=False)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(df['n'], df['Robustness score'], marker='o')
    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Robustness Score (MMD)')
    plt.title('Robustness Score vs Number of Samples')
    plt.grid(True)
    plt.savefig('robustness_scores_plot.png')
    plt.show()

print("Completed...!!!")
