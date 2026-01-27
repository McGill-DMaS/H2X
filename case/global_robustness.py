import json
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Function to compute MMD score between two sets of data
def compute_mmd(x, y, sigma):
    kxx = rbf_kernel(x, x, gamma=1.0 / (2.0 * sigma ** 2))
    kxy = rbf_kernel(x, y, gamma=1.0 / (2.0 * sigma ** 2))
    kyy = rbf_kernel(y, y, gamma=1.0 / (2.0 * sigma ** 2))
    mmd = np.mean(kxx) - 2.0 * np.mean(kxy) + np.mean(kyy)
    return 1-mmd

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

# Function to process data for LEMNA and AttNN
def filter_keys(data):
    return {key: value for key, value in data.items() if all(op not in key for op in ['<=', '<', '>', '>='])}

def process_data(json_data):
    benign_data = [filter_keys(sample['data']) for sample in json_data['result_benign']]
    malware_data = [filter_keys(sample['data']) for sample in json_data['result_malware']]
    return benign_data, malware_data

def to_array(data_list):
    keys = list(data_list[0].keys())
    data_array = np.array(
        [[np.mean(sample[key]) if isinstance(sample[key], list) else sample[key] for key in keys] for sample in
         data_list])
    return data_array

# Function to calculate MMD for different values of n
def calculate_mmd_for_n_values(json_data, n_values, sigma=1.0, method='shap'):
    mmd_scores = []

    for n in n_values:
        if method in ['H2X', 'SHAP']:
            shap_benign = json_data['result_benign'][:n]
            shap_malware = json_data['result_malware'][:n]

            df_benign_n = [convert_to_flat_dict(sample['data']) for sample in shap_benign]
            df_malware_n = [convert_to_flat_dict(sample['data']) for sample in shap_malware]

            df_benign_n_com_list = [list(sample.values()) for sample in df_benign_n]
            df_malware_n_com_list = [list(sample.values()) for sample in df_malware_n]

            df_benign_n_com_list = [np.concatenate(sample).tolist() for sample in df_benign_n_com_list]
            df_malware_n_com_list = [np.concatenate(sample).tolist() for sample in df_malware_n_com_list]

        else:
            benign_data, malware_data = process_data(json_data)
            df_benign_n_com_list = to_array(benign_data[:n])
            df_malware_n_com_list = to_array(malware_data[:n])

        df_benign_n_com_list = np.array(df_benign_n_com_list)
        df_malware_n_com_list = np.array(df_malware_n_com_list)

        df_benign_n_com_list[np.isinf(df_benign_n_com_list)] = np.nan
        df_malware_n_com_list[np.isinf(df_malware_n_com_list)] = np.nan

        df_benign_n_com_list = pd.DataFrame(df_benign_n_com_list).dropna(axis=1).values
        df_malware_n_com_list = pd.DataFrame(df_malware_n_com_list).dropna(axis=1).values

        scaler = StandardScaler()
        all_data = np.vstack((df_benign_n_com_list, df_malware_n_com_list))
        all_data = scaler.fit_transform(all_data)
        df_benign_n_com_list = all_data[:len(df_benign_n_com_list)]
        df_malware_n_com_list = all_data[len(df_benign_n_com_list):]

        mmd = compute_mmd(df_benign_n_com_list, df_malware_n_com_list, sigma)
        mmd_scores.append((n, mmd))
        print(f'MMD for {method} n={n}: {mmd}')

    return mmd_scores

if __name__ == "__main__":
    n_values = list(range(10, 1010, 10))
    sigma = 1.0

    explanation_files = {
        'H2X': 'result/H2X_explanations_20240704.json',
        'SHAP': '../SOTA/SHAP/result/SHAP_explanations.json',
        'LEMNA': '../SOTA/LEMNA/result/LEMNA_explanations.json',
        'AttNN': '../SOTA/AttNN/result/AttNN_explanations.json'
    }

    all_scores = {}

    for method, filepath in explanation_files.items():
        with open(filepath, 'r') as f:
            json_data = json.load(f)

        method_scores = calculate_mmd_for_n_values(json_data, n_values, sigma, method=method)
        all_scores[method] = [score[1] for score in method_scores]

    df = pd.DataFrame({'n': n_values})
    for method in all_scores:
        df[f'{method}_robustness_score'] = all_scores[method]

    df.to_csv('result/robustness_scores_comparison_v2_r.csv', index=False)

    plt.figure(figsize=(10, 6))
    for method in all_scores:
        plt.plot(df['n'], df[f'{method}_robustness_score'], marker='o', label=method)

    plt.xlabel('Number of Samples (n)')
    plt.ylabel('Robustness Score (MMD)')
    plt.title('Robustness Score vs Number of Samples')
    plt.grid(True)
    plt.legend()
    plt.savefig('result/robustness_scores_comparison_plot.png')
    plt.show()

print("Completed...!!!")
