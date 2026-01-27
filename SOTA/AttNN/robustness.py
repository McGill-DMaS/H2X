import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def compute_mmd(x, y, sigma):
    """
    Computes the Maximum Mean Discrepancy (MMD) between two sets of data.
    """
    kxx = rbf_kernel(x, x, gamma=1.0 / (2.0 * sigma ** 2))
    kxy = rbf_kernel(x, y, gamma=1.0 / (2.0 * sigma ** 2))
    kyy = rbf_kernel(y, y, gamma=1.0 / (2.0 * sigma ** 2))
    mmd = np.mean(kxx) - 2.0 * np.mean(kxy) + np.mean(kyy)
    return mmd


def filter_keys(data):
    """
    Filters out keys that contain conditions such as '<=' or similar.
    """
    return {key: value for key, value in data.items() if all(op not in key for op in ['<=', '<', '>', '>='])}


def process_data(json_data):
    """
    Processes the JSON data to extract the feature data for benign and malware samples.
    """
    benign_data = [filter_keys(sample['data']) for sample in json_data['result_benign']]
    malware_data = [filter_keys(sample['data']) for sample in json_data['result_malware']]
    return benign_data, malware_data


def to_array(data_list):
    """
    Converts a list of dictionaries to a 2D numpy array.
    """
    keys = list(data_list[0].keys())
    data_array = np.array(
        [[np.mean(sample[key]) if isinstance(sample[key], list) else sample[key] for key in keys] for sample in
         data_list])
    return data_array


def calculate_mmd_for_n_values(json_data, n_values, sigma=1.0):
    """
    Calculate MMD for different values of n and return the results.
    """
    mmd_scores = []
    for n in n_values:
        benign_data, malware_data = process_data(json_data)
        benign_array = to_array(benign_data[:n])
        malware_array = to_array(malware_data[:n])

        # Standardize the data
        scaler = StandardScaler()
        all_data = np.vstack((benign_array, malware_array))
        all_data = scaler.fit_transform(all_data)
        benign_array = all_data[:len(benign_array)]
        malware_array = all_data[len(benign_array):]

        mmd = compute_mmd(benign_array, malware_array, sigma)
        mmd_scores.append((n, mmd))
        print(f'MMD for n={n}: {mmd}')
    return mmd_scores


if __name__ == "__main__":
    # Load the JSON data
    with open('result/attention_explanations.json', 'r') as f:
        json_data = json.load(f)

    # Define the range of n values
    n_values = list(range(10, 1010, 10))
    sigma = 1.0

    # Calculate MMD for different values of n
    mmd_scores = calculate_mmd_for_n_values(json_data, n_values, sigma)

    # Save the results in a CSV file
    df = pd.DataFrame(mmd_scores, columns=['n', 'Robustness score'])
    df.to_csv('result/AttNN_robustness_scores.csv', index=False)

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
