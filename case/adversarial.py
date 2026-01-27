from model.predict import *


#train_csv_url = "C:\\Users\\Saqib\\PycharmProjects\\MalD\\model\\dataset\\train.csv"
df = get_trainDF()


# Get indices of all male instances
benign_indices = df[df['label'] == 'benign'].index.tolist()

# Create empty list to store new female datasets
advers_malware_datasets = []
size_of_advers_malware_datasets = 100

# Loop over 10 times to create 10 new female datasets
for i in range(size_of_advers_malware_datasets):
    # Randomly choose one male instance
    benign_instance_index = np.random.choice(benign_indices)
    benign_instance = df.loc[benign_instance_index]
    # Get values of columns 'cr', 'cr+1', ..., 'cr+m' of the chosen male instance
    columns_to_add = list(benign_instance.keys())[benign_instance.keys().get_loc('f_garbage_0'):benign_instance.keys().get_loc('f_ExportsList_other_15')+1]
    column_values = benign_instance[columns_to_add]
    # Filter female instances and choose one randomly
    malware_instances = df[df['label'] == 'malware']
    malware_instance_index = malware_instances.iloc[[i]]

    # Create new female instance by adding column_values to the chosen female instance
    advers_malware_instance = malware_instance_index.copy()
    advers_malware_instance[columns_to_add] = (advers_malware_instance[columns_to_add] +  (column_values))/2
    x_temp, y_temp = dataTranformerSeprator(advers_malware_instance)
    y_pred = predict(x_temp)
    print(i, " : ", y_pred)
    # Add new female instance to list of new female datasets
    advers_malware_datasets.append(advers_malware_instance)
"""
advers_malware_datasets = np.array(advers_malware_datasets).reshape(30,472)
pd.DataFrame(advers_malware_datasets).to_csv("advers_malware_datasets.csv")
"""

"""
This case study is not successful should be consider later
"""
