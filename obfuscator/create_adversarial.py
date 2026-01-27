import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import joblib
import json
from feature_not_change import return_list

# Load the dataset
dataset = pd.read_csv("../model/dataset/test_20240704.csv")

dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
dataset.drop_duplicates(inplace=True)
dataset['label'] = dataset['label'].str.replace(r'^benign_.+', 'benign', regex=True)

# Filter out classes with few samples
value_counts = dataset['label'].value_counts()
to_remove = value_counts[value_counts <= 50].index
dataset = dataset[~dataset['label'].isin(to_remove)]
dataset = dataset.drop(['filename'], axis=1)
2
# Split the data into training and test sets

# Load the trained model and scaler
output_path = "../model/trained_model/model_20240730_024716"
MODEL_PATH = f'{output_path}'
SCALER_PATH = f'{output_path}/scaler.pkl'
LABEL_ENCODER_PATH = f'{output_path}/label_encoder.pkl'

model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)


# Identify key system calls for benign samples
benign_samples = dataset[dataset['label'] == 'benign']
benign_samples = benign_samples.drop(columns=['label'])
benign_mean_freq = benign_samples.mean()

important_features_list = return_list()

# Function to create adversarial examples
def create_adversarial_example(sample, class_label, benign_mean_freq, important_features, increase_factor=2.5, max_attempts=100):
    adversarial_sample = sample.copy()
    changed_features = []

    for attempt in range(max_attempts):
        for feature in benign_mean_freq.index:
            if feature not in important_features:
                adversarial_sample[feature] = min(adversarial_sample[feature] * increase_factor,
                                                  benign_mean_freq[feature] * increase_factor)
                if feature not in changed_features:
                    changed_features.append(feature)

        # Scale the adversarial sample
        adversarial_sample_scaled = scaler.transform(adversarial_sample.drop(['label']).values.reshape(1, -1))

        # Predict using the model
        predicted_prob = model.predict(adversarial_sample_scaled)
        predicted_label = label_encoder.inverse_transform([np.argmax(predicted_prob)])[0]
        print(class_label, predicted_label)
        # Check if misclassified as benign
        if predicted_label == 'benign':
            print(f"Adversarial example created after {attempt + 1} attempts.")
            return adversarial_sample, changed_features
        else:
            continue
    return None, changed_features

# Function to generate and save adversarial examples
def generate_and_save_adversarial_examples():
    adversarial_samples = []
    #for class_label in dataset['label'].unique():
    for class_label in [label for label in dataset['label'].unique() if label != "benign"]:
        class_samples = dataset[dataset['label'] == class_label]
        if len(class_samples) == 100:
            continue
        selected_samples = class_samples.sample(min(1000, len(class_samples)))

        important_features = return_list()

        for _, sample in selected_samples.iterrows():
            adversarial_sample, changed_features = create_adversarial_example(sample, class_label, benign_mean_freq, important_features)
            sample_dict = {}
            if adversarial_sample is not None:
                sample_dict['orignal_index'] = sample.name
                sample_dict['orignal_label'] = class_label
                sample_dict['orignal_sample'] = sample.to_dict()
                sample_dict['adversarial_sample'] = adversarial_sample.to_dict()
                sample_dict['changed_features'] = changed_features
                print(sample_dict)
                adversarial_samples.append(sample_dict)

    # Save the adversarial samples
    with open('result/adversarial_samples_.json', 'w') as f:
        json.dump(adversarial_samples, f, indent=4)

generate_and_save_adversarial_examples()
