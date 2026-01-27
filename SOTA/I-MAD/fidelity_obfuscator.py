import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib

def load_explanations(json_file):
    with open(json_file, 'r') as f:
        explanations = json.load(f)
    return explanations

def extract_top_k_features(data, k):
    feature_importance = {}
    for item in data:
        for key, value in item['data'].items():
            feature_name = key.split(' ')[0]  # Generalize feature names
            if feature_name not in feature_importance:
                feature_importance[feature_name] = 0
            feature_importance[feature_name] += abs(value)

    # Sort features by their importance and select top k
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:k]
    top_k_features = [feature[0] for feature in sorted_features]
    return top_k_features

def prepare_data(explanations, top_k_features, result_type='result_malware'):
    data = []
    labels = []
    for item in explanations[result_type]:
        features = item['data']
        feature_vector = []
        for feature in top_k_features:
            # Check if the feature exists in the explanation, else use 0
            feature_value = next((value for key, value in features.items() if key.startswith(feature)), 0)
            feature_vector.append(feature_value)
        data.append(feature_vector)
        labels.append(1 if result_type == 'result_malware' else 0)
    return np.array(data), np.array(labels)

def load_surrogate_model():
    model_path = 'result/surrogate_model.pkl'
    model = joblib.load(model_path)
    return model

def calculate_fidelity(original_predictions, surrogate_predictions):
    fidelity_score = accuracy_score(original_predictions, surrogate_predictions)
    return fidelity_score

# Load the LIME explanations
explanations = load_explanations('result/iffnn_explanations_obfuscator.json')

# Extract top k features from malware explanations
top_k_features = extract_top_k_features(explanations['result_malware'], k=4)

# Prepare data for training the surrogate model
#X_malware, y_malware = prepare_data(explanations, top_k_features, result_type='result_malware')
X_benign, y_benign = prepare_data(explanations, top_k_features, result_type='result_benign')

# Combine malware and benign data
X = np.concatenate([X_benign], axis=0)
y = np.concatenate([y_benign], axis=0)

# Normalize the features
scaler_path = 'result/scaler.pkl'
scaler = joblib.load(scaler_path)
X = scaler.fit_transform(X)

# Train the surrogate model
surrogate_model = load_surrogate_model()

# Get the original model predictions (1 for malware, 0 for benign)
original_predictions = [0 for _ in y]

# Get the surrogate model predictions
surrogate_predictions = surrogate_model.predict(X)

# Calculate fidelity score
fidelity_score = calculate_fidelity(original_predictions, surrogate_predictions)
print('Fidelity Score:', fidelity_score)

