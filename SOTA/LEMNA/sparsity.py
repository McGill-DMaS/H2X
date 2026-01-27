import json
import pandas as pd
import re
import numpy as np

def generalize_key(key):
    """
    Generalizes the key by removing conditions such as '<=', '<', '>', '>='.
    """
    return re.sub(r' <=.*| <.*| >.*| >=.*', '', key)

def get_ImportantFeatures_for_being_malicious(df_malware, n_features):
    # Extract important features for being malicious
    important_features = [key for key, value in df_malware.items() if value >= np.mean(list(df_malware.values()))]
    return important_features

# Load the LIME explanation JSON file
with open('result/LEMNA_explanations_obfuscator.json', 'r') as f:
    lime_ex = json.load(f)

def get_sparsityForN(n, lime_ex):
    # Extract the local explanations for malware samples
    shap_malicious = lime_ex['result_malware']

    spars_temp = []
    for i in range(n):
        df_malware = shap_malicious[i]["data"]
        # Generalize the keys
        print(df_malware.items())
        df_malware = {generalize_key(key): value for key, value in df_malware.items()}
        print(df_malware)
        # Get the important features for being malicious
        top_hybrid = get_ImportantFeatures_for_being_malicious(df_malware, len(df_malware))

        # Calculate sparsity
        spars_temp.append(1 - (len(top_hybrid) / len(df_malware)))

    return spars_temp

if __name__ == "__main__":
    spars_dict = {}
    n_files = 500
    spars = get_sparsityForN(n_files, lime_ex)
    for it_ in range(n_files):
        total_spars = 0
        for i in range(it_ + 1):
            total_spars += spars[i]
        spars_dict[it_] = (total_spars / (it_ + 1))

    print(spars_dict)
    df = pd.DataFrame(list(spars_dict.items()), columns=['Key', 'Value'])

    # Save the dataframe to a CSV file
    df.to_csv('result/LEMNA_sparsity_obfuscator.csv', index=False)
