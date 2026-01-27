import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import time
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

# Load and preprocess the dataset
test_file_url = '../../model/dataset/test_20240809.csv'
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
model.load_state_dict(torch.load('result/iffnn_model_v2.pth'))
model.eval()

# Calculate evaluation metrics on test data
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = torch.argmax(y_pred, dim=1).numpy()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Measure throughput for processing files and save results to CSV
def measure_throughput_and_save_csv(file_sizes):
    throughput_data = []

    for n_files in file_sizes:
        start_time = time.time()
        for _ in range(n_files):
            with torch.no_grad():
                _ = model(X_test)
        end_time = time.time()

        # Calculate time taken and throughput
        total_time_taken = end_time - start_time
        avg_time_per_file = total_time_taken / n_files if n_files > 0 else 0
        throughput_data.append([n_files, total_time_taken, avg_time_per_file])

        print(f'Throughput for {n_files} files: {n_files / total_time_taken:.2f} files/second')

    # Convert to DataFrame and save as CSV
    throughput_df = pd.DataFrame(throughput_data, columns=['file_size', 'total_time_taken', 'average_time_taken'])
    throughput_df.to_csv('result/iffnn_throughput_results.csv', index=False)

# Run throughput measurement for file sizes 1-1000
file_sizes = list(range(1, 1001))
measure_throughput_and_save_csv(file_sizes)
