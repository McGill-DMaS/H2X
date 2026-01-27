import json
from model.local_ex import get_LocalExplanationFile
from model.hybrid_ex import *
from model.hybrid_ex_cca import get_ExplanationMetric, get_CORR_cal

#get sample
with open('../model/dataset/temp_data_micro/Ex_Features_origin_16437__.file.gz.json', 'r') as f:
    feature_file = json.load(f)

#get local
shap_values = get_LocalExplanationFile('origin_16437__.file.gz')

#get global
with open('../model/explaination/global_ex.json', 'r') as f:
    weights = json.load(f)

n = len(shap_values.keys())
corr = get_CORR_cal(weights, shap_values)
print(corr)
get_hybrid = get_hybridImportantFeatures(weights, shap_values, n)
final_ex = {}
for key in get_hybrid:
    common_key = key.replace('f_', '').replace('_x', '').replace('_0', '')
    final_ex[key] = feature_file[common_key]
with open("../model/dataset/temp_data_micro/final_ex_origin_16437__.file.gz.json", "w") as fp:
    json.dump(final_ex, fp)
#weighted_local = get_weightedLocal(weights, shap_values)
#cca_metrics = get_ExplanationMetric(weights, shap_values, n)
#sorted_dict = dict(sorted(top_features.items(), key=lambda x: x[1][1], reverse=True))

#print(sorted_dict)
print("Completed...!!!")
