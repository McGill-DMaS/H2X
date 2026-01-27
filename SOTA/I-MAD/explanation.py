import torch
import torch.nn as nn
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from collections import OrderedDict
import numpy as np


# Define IFFNN model
class IFFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, bicls=False, use_dropout=False, act_func='relu'):
        super(IFFNN, self).__init__()
        self.input_size = input_size
        self.bicls = bicls
        self.num_classes = num_classes
        dic = OrderedDict()
        previous_dim = input_size
        for i, dim in enumerate(hidden_sizes):
            lay = nn.Linear(previous_dim, dim)
            previous_dim = dim
            dic['linear' + str(i)] = lay
            if act_func == 'tanh':
                dic['act_func' + str(i)] = nn.Tanh()
            else:
                assert (act_func == 'relu')
                dic['act_func' + str(i)] = nn.ReLU()

        n_hid = len(hidden_sizes)
        if bicls:
            lay = nn.Linear(previous_dim, input_size)
            self.last_bias = torch.nn.Parameter(torch.zeros([1]))
        else:
            lay = nn.Linear(previous_dim, input_size * num_classes)
            self.last_bias = torch.nn.Parameter(torch.zeros([num_classes]))

        dic['linear' + str(n_hid)] = lay
        self.iffnnpart1 = nn.Sequential(dic)

        self.register_parameter(name='bias', param=self.last_bias)

    def forward(self, x):
        out = self.iffnnpart1(x)
        if self.bicls:
            full_features = x
        else:
            full_features = x.repeat(1, self.num_classes)
        out = full_features * out
        if self.bicls:
            out = out.sum(axis=1)
        else:
            out = out.reshape(-1, self.num_classes, self.input_size)
            out = out.sum(axis=2)
        out = out + self.last_bias
        return out

    def explain(self, x):
        out = self.iffnnpart1(x)
        if self.bicls:
            full_features = x
        else:
            full_features = x.repeat(1, self.num_classes)
        out = full_features * out
        if not self.bicls:
            out = out.reshape(-1, self.num_classes, self.input_size)
        out = out.cpu().detach().numpy()
        x = x.cpu().detach().numpy()
        if self.bicls:
            results = explain_binary(x, out)
        else:
            results = explain_multi(x, out, self.num_classes)
        return results


def explain_multi(x, out, num_classes, n_channel=1):
    results = []
    for i in range(len(x)):
        current = []
        sam = x[i]
        current.append(sam)
        for j in range(num_classes):
            weight = out[i][j]
            if n_channel == 1:
                current.append(weight)
            else:
                weight = weight.reshape(n_channel, -1)
                current.append(weight)
        results.append(current)
    return results


def explain_binary(x, out, n_channel=1):
    results = []
    for i in range(len(x)):
        current = []
        sam = x[i]
        current.append(sam)
        weight = out[i]
        if n_channel != 1:
            weight = weight.reshape(n_channel, -1)
        current.append(-weight)
        current.append(weight)
        results.append(current)
    return results


# Load and preprocess the dataset
#test_file_url = '../../model/dataset/test_20240704.csv'
test_file_url = '../../obfuscator/result/adversarial_samples.csv'
df_test = pd.read_csv(test_file_url)

# Split features and labels
y_test = df_test['label']
X_test = df_test.drop(labels=['label', 'filename'], axis=1)

# Encode labels
le = LabelEncoder()
y_test = le.fit_transform(y_test)

# Standardize features
sc = StandardScaler()
X_test = sc.fit_transform(X_test)

# Convert to PyTorch tensors
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Model parameters
input_size = X_test.shape[1]
hidden_sizes = [64, 32]
num_classes = len(np.unique(y_test))

# Initialize and load the model
model = IFFNN(input_size, hidden_sizes, num_classes, bicls=False)
model.load_state_dict(torch.load('result/iffnn_model.pth'))
model.eval()

# Generate explanations for the test data
explanations = model.explain(X_test)

# Process explanations to match the required output format
explanation_results = {
    "result_benign": [],
    "result_malware": []
}

# Placeholder filenames for demonstration; replace with actual filenames if available
filenames = df_test['filename'].values

for i, (filename, exp) in enumerate(zip(filenames, explanations)):
    values = exp[0]
    positive_weights = exp[2]
    negative_weights = exp[1]

    result = values * (positive_weights - negative_weights)

    data = {f"feature_{j}": float(result[j]) for j in range(len(result))}

    entry = {
        "file": str(filename),
        "data": data
    }

    # Assuming labels 0 and 1 represent benign and malware respectively
    if y_test[i] == 0:
        explanation_results["result_benign"].append(entry)
    else:
        explanation_results["result_malware"].append(entry)

# Save the explanations to a JSON file
with open('result/iffnn_explanations_obfuscator.json', 'w') as f:
    json.dump(explanation_results, f, indent=4)

print("Explanations saved to explanations.json")
