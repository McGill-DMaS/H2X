import json
import pandas as pd


# Function to determine important features for being malicious
def get_ImportantFeatures_for_being_malicious(df_malware, n_features):
    important_features = []
    for feature, values in df_malware.items():
        if values[1] > values[0]:  # Support for being malicious > support for benign
            important_features.append(feature)
    return important_features


# Function to calculate sparsity for N samples
def get_sparsityForN(n, shap_ex):
    shap_malicious = shap_ex["result_malware"][:n]
    spars_temp = []

    for i in range(n):
        df_malware = shap_malicious[i]["data"]
        top_hybrid = get_ImportantFeatures_for_being_malicious(df_malware, len(df_malware))
        spars_temp.append(1 - (len(top_hybrid) / len(df_malware)))

    return spars_temp


if __name__ == "__main__":
    # Load SHAP explanation JSON file
    with open('result/SHAP_explanations_obfuscator.json', 'r') as f:
        shap_ex = json.load(f)

    # Specify the number of files to analyze
    n_files = 500
    spars = get_sparsityForN(n_files, shap_ex)

    # Calculate cumulative sparsity
    spars_dict = {}
    for it_ in range(n_files):
        total_spars = 0
        for i in range(it_ + 1):
            total_spars += spars[i]
        spars_dict[it_] = (total_spars / (it_ + 1))

    # Print sparsity results
    print(spars_dict)

    # Save the results to a CSV file
    df = pd.DataFrame(list(spars_dict.items()), columns=['Key', 'Value'])
    df.to_csv('result/SHAP_sparsity_obfuscator.csv', index=False)

    print("Completed...!!!")
