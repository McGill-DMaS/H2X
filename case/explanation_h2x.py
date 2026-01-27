import json
from model.local_ex import get_LocalExplanationFile, get_nLabeledLocalExplanation
from model.hybrid_ex import *
from model.hybrid_ex_cca import get_ExplanationMetric, get_CORR_cal


#get local of 10 mal, and benign
n = 600
shap_malicious, shap_benign = get_nLabeledLocalExplanation(n)

#get global
with open('../model/explaination/global_ex_20240704.json', 'r') as f:
    weights = json.load(f)

res = {}
res["result_benign"] = []
res["result_malware"] = []
for i in range(n):
    df_malware = {}
    df_benign = {}

    df_malware['shap'] = shap_malicious[i]["data"]
    df_benign['shap'] = shap_benign[i]["data"]

    df_malware['cca_wv'] = get_ExplanationMetric(weights, df_malware['shap'], len(df_malware['shap']))
    df_benign['cca_wv'] = get_ExplanationMetric(weights, df_benign['shap'], len(df_benign['shap']))
    row_malware = {'file': i, 'data': df_malware['cca_wv']}
    row_benign = {'file': i, 'data': df_benign['cca_wv']}
    res["result_benign"].append(row_benign)
    res["result_malware"].append(row_malware)

json_file_path = f'result/H2X_explanations_obfuscator.json'
with open(json_file_path, 'w') as json_file:
    json.dump(res, json_file, indent=4)

print("Completed...!!!")
