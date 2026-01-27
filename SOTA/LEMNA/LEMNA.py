# libraries
# ----------------------------------------------------
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import tensorflow as tf
from sklearn.mixture import GaussianMixture
import numpy as np

module_dir = os.path.dirname(__file__)
model_file = os.path.join(module_dir, '../../model/trained_model', 'model_20240730_024716')
train_file_url = os.path.join(module_dir, '../../model/dataset', 'train_20240704.csv')
#test_file_url = os.path.join(module_dir, '../../model/dataset', 'test_20240704.csv')
test_file_url = os.path.join(module_dir, '../../obfuscator/result', 'adversarial_samples.csv')
model = tf.keras.models.load_model(model_file)

def avg_contributor_keys_lemna(d):
    result = {}
    key_patterns = ['f_longWord_', 'f_specialKeyword_', 'f_ipaddresses_',
                    'f_sentences_', 'f_ImportsList_create_', 'f_ImportsList_resume_',
                    'f_ImportsList_kill_', 'f_ImportsList_call_', 'f_ImportsList_delete_',
                    'f_ImportsList_other_', 'f_ExportsList_open_', 'f_ExportsList_close_', 'f_ExportsList_create_',
                    'f_ExportsList_call_', 'f_ExportsList_delete_',
                    'f_longWord_', 'f_inValEmails_', 'f_emails_', 'f_DIRs_', 'f_fileName_', 'f_garbage_', 'f_URLs_',
                    'f_ImportsList_open_', 'f_ImportsList_close_', 'f_ExportsList_resume_', 'f_ExportsList_kill_',
                    'f_ExportsList_other_']

    for pattern in key_patterns:
        keys = [key for key in d.keys() if pattern in key]
        if len(keys) > 0:
            collected_list = [d[key] for key in keys if isinstance(d[key], float)]
            if collected_list:
                avg = sum(collected_list) / len(collected_list)
                result[pattern + 'x'] = avg
        for key in keys:
            d.pop(key, None)
    result.update(d)
    return result

def get_LocalExplanation(train_file_url, test_file_url, model, start_index):
    end_index = start_index + 1
    df_train = pd.read_csv(train_file_url)
    y_train = df_train['label']
    x_train = df_train.drop(labels=['label', 'filename'], axis=1)

    ohe = OneHotEncoder()
    le = LabelEncoder()
    cols = x_train.columns.values
    for col in cols:
        x_train[col] = le.fit_transform(x_train[col])
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    y_train = le.fit_transform(y_train)

    df_test = pd.read_csv(test_file_url)
    y_test = df_test['label']
    x_test = df_test.drop(labels=['label', 'filename'], axis=1)
    cols = x_test.columns.values
    for col in cols:
        x_test[col] = le.fit_transform(x_test[col])
    sc = StandardScaler()
    x_test = sc.fit_transform(x_test)
    y_test = le.fit_transform(y_test)

    # Create LEMNA explanation
    sample = x_test[start_index]
    a = data_inverse(x_train, sample, 1000)

    gm = GaussianMixture(n_components=len(np.unique(y_train)), covariance_type='diag')
    p_data_train = a[1]
    p_data_test = model.predict(p_data_train).argmax(axis=1)
    gm.fit(p_data_train, p_data_test)

    components = gm.precisions_
    target = gm.predict(sample.reshape(1, -1))[0]

    ind = []
    for i in range(len(p_data_train)):
        if gm.predict(p_data_train[i].reshape(1, -1))[0] == target:
            ind.append(i)

    feature_count = np.zeros(x_train.shape[1])
    for i in ind:
        feature_count += (p_data_train[i] != 0).astype(int)

    feature_importance = feature_count / np.sum(feature_count)

    explanation_dict = {col: importance for col, importance in zip(df_train.columns[:-2], feature_importance)}
    key_features_dict = avg_contributor_keys_lemna(explanation_dict)

    file_name = str(df_test["filename"][start_index])
    return key_features_dict, file_name

def data_inverse(x_train, data_row, num_samples):
    data = np.zeros((num_samples, data_row.shape[0]))
    for i in range(num_samples):
        data[i] = x_train[np.random.choice(range(x_train.shape[0]))]
    data[0] = data_row
    return data, data

def get_nLocalExplanation(n):
    model_PATH = "../../model/trained_model/model_20240730_024716"
    model = tf.keras.models.load_model(model_PATH)
    train_file_url = '../../model/dataset/train_20240704.csv'
    #test_file_url = '../../model/dataset/test_20240704.csv'
    test_file_url = '../../obfuscator/result/adversarial_samples.csv'

    result = []
    for i in range(n):
        temp_dict = {}
        start_index = i
        key_features_dict, file_name = get_LocalExplanation(train_file_url, test_file_url, model, start_index)
        temp_dict["file"] = file_name
        temp_dict["data"] = key_features_dict
        result.append(temp_dict)

    return result

def get_LocalExplanationFile(file_name):
    df_test = pd.read_csv(test_file_url)

    index = df_test.index[df_test['filename'] == file_name][0]
    key_features_dict, file_name = get_LocalExplanation(train_file_url, test_file_url, model, index)

    return key_features_dict

def get_nLabeledLocalExplanation(n):
    df_test = pd.read_csv(test_file_url)

    df_benign = df_test[df_test['label'] == 'benign']
    df_malware = df_test[df_test['label'] == 'malware']

    result_benign = []
    result_malware = []
    benign_files = df_benign.sample(n=n)['filename'].tolist()
    malware_files = df_malware.sample(n=n)['filename'].tolist()

    for file_ in benign_files:
        key_features_dict = get_LocalExplanationFile(file_)
        result_benign.append({'file': file_, 'data': key_features_dict})

    for file_ in malware_files:
        key_features_dict = get_LocalExplanationFile(file_)
        result_malware.append({'file': file_, 'data': key_features_dict})

    return result_benign, result_malware

if __name__ == "__main__":
    d1, d2 = get_nLabeledLocalExplanation(n=600)

    all_results = {"result_benign": d1, "result_malware": d2}
    json_file_path = f'result/LEMNA_explanations_obfuscator.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

print("Completed...!!!")
