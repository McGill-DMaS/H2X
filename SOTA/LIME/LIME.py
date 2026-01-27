# libraries
# ----------------------------------------------------
import json
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from lime.lime_tabular import LimeTabularExplainer

module_dir = os.path.dirname(__file__)
model_file = os.path.join(module_dir, '../../model/trained_model', 'model_20240730_024716')
train_file_url = os.path.join(module_dir, '../../model/dataset', 'train_20240704.csv')
#test_file_url = os.path.join(module_dir, '../../model/dataset', 'test_20240704.csv')
test_file_url = os.path.join(module_dir, '../../obfuscator/result', 'adversarial_samples.csv')
model = tf.keras.models.load_model(model_file)

def avg_contributor_keys_lime(d):
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
        #keys = [key for key in d.keys() if key.startswith(pattern)]
        keys = [key for key in d.keys() if pattern in key]  # Change here
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
    # split target and features
    y_train = df_train['label']
    x_train = df_train.drop(labels=['label', 'filename'], axis=1)

    # encoding
    ohe = OneHotEncoder()
    le = LabelEncoder()
    cols = x_train.columns.values
    for col in cols:
        x_train[col] = le.fit_transform(x_train[col])
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    y_train = le.fit_transform(y_train)

    # testing data to test local ex
    df_test = pd.read_csv(test_file_url)
    # split target and features
    y_test = df_test['label']
    x_test = df_test.drop(labels=['label', 'filename'], axis=1)
    cols = x_test.columns.values
    for col in cols:
        x_test[col] = le.fit_transform(x_test[col])
    sc = StandardScaler()
    x_test = sc.fit_transform(x_test)
    y_test = le.fit_transform(y_test)

    # Use LIME to explain the predictions of the model
    explainer = LimeTabularExplainer(x_train, feature_names=df_train.columns[:-2], class_names=['benign', 'malware'], discretize_continuous=True)
    exp = explainer.explain_instance(x_test[start_index], model.predict, num_features=len(df_train.columns[:-2]))

    # Get the top n key features for the first class of the first instance
    explanation_dict = {item[0]: item[1] for item in exp.as_list()}
    key_features_dict = avg_contributor_keys_lime(explanation_dict)

    file_name = str(df_test["filename"][start_index])
    return key_features_dict, file_name

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

    # Filter out benign and malware files separately
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

    # Save the results as JSON
    all_results = {"result_benign": d1, "result_malware": d2}
    json_file_path = f'result/LIME_explanations_obfuscator.json'
    with open(json_file_path, 'w') as json_file:
        json.dump(all_results, json_file, indent=4)

print("Completed...!!!")
