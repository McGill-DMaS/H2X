import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from collections import OrderedDict
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import time

# Ensure reproducibility
random_seed = 0
torch.manual_seed(random_seed)

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
train_file_url = '../../model/dataset/train_20240809.csv'
test_file_url = '../../model/dataset/test_20240809.csv'

df_train = pd.read_csv(train_file_url)
df_test = pd.read_csv(test_file_url)

# Split features and labels
y_train = df_train['label']
X_train = df_train.drop(labels=['label', 'filename'], axis=1)
y_test = df_test['label']
X_test = df_test.drop(labels=['label', 'filename'], axis=1)

# Encode labels
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Standardize features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Model parameters
input_size = X_train.shape[1]
hidden_sizes = [64, 32]
num_classes = len(np.unique(y_train))

# Initialize and train the model
model = IFFNN(input_size, hidden_sizes, num_classes, bicls=False)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Save the model
torch.save(model.state_dict(), 'result/iffnn_model_v2.pth')

# Calculate evaluation metrics on test data
model.eval()
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

