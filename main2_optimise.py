import os
import pandas as pd
import gzip
from getNumericFeatures import *
from getStringFeatures import *
from ClusterToVec import *
from getImportExportFeatures import *
import gc

def createPandaInitialTable(featureSize):
    file = "for_setup"
    data = getFeaturesVec(file, featureSize)
    columns = []
    for key in data.keys():
        len_key = len(data[key]) if isinstance(data[key], (list, tuple)) else 1
        columns += ['f_' + key + '_' + str(i) for i in range(len_key)]
    columns += ['filename', 'label']
    csv_data = pd.DataFrame(columns=columns)
    return csv_data, columns

def appendCSV(data_list, data, filename, label):
    new_row = []
    for key in data.keys():
        if isinstance(data[key], (list, tuple)):
            new_row.extend(data[key])
        else:
            new_row.append(data[key])
    new_row.extend([filename, label])
    data_list.append(new_row)

def getFeaturesVec(file, featureSize):
    numericFeatures = getNumericFeatures(file)
    stringFeatures, importExportList = getStrings(file)
    importExportListFeatures = getImportExportCluster(importExportList)
    stringFeaturesVec = getClusterToVec(stringFeatures, featureSize)
    importExportFeaturesVec = getClusterToVec(importExportListFeatures, featureSize)
    data = {**numericFeatures, **stringFeaturesVec, **importExportFeaturesVec}
    return data

# def processZIP(zipf, featureSize):
#     tempf = 'for_setup'
#     with open(tempf, 'wb') as f_out, gzip.open(zipf, 'rb') as f_in:
#         f_out.writelines(f_in)
#     data = getFeaturesVec(tempf, featureSize)
#     #os.remove(tempf)
#     return data

def processZIP(zipf, featureSize):
    tempf = 'for_temp_malware_0'
    try:
        with gzip.open(zipf, 'rb') as f_in:
            if f_in.peek(1):  # Check if file is not empty
                with open(tempf, 'wb') as f_out:
                    f_out.writelines(f_in)
                data = getFeaturesVec(tempf, featureSize)
            else:
                print(f"File {zipf} is empty.")
                data = {}
    except Exception as e:
        print(f"Error processing zip file {zipf}: {e}")
        data = {}
    return data


def check_all_files(dir, folder, featureSize):
    path = os.path.join(dir, folder)
    current_files = os.listdir(path)
    no_f = len(current_files)
    data_list = []

    for countf, f in enumerate(current_files, 1):
        try:
            zipf = os.path.join(path, f)
            if f.endswith('.gz'):
                data = processZIP(zipf, featureSize)
            elif "." not in f:
                data = getFeaturesVec(zipf, featureSize)
            else:
                continue
            appendCSV(data_list, data, f, folder)  # folder name is going to be label
            print(f"{countf} / {no_f} done!")
        except Exception as e:
            print(f"Error processing file {f}: {e}")
            continue

    if data_list:
        columns = createPandaInitialTable(featureSize)[1]
        csv_data = pd.DataFrame(data_list, columns=columns)
        foutput = f"output/csv_data_output_{folder}.csv"
        csv_data.to_csv(foutput, index=False)
        print("Data saved to:", foutput)
        del csv_data
        gc.collect()

# convert data to csv data
featureSize = 16
dir = 'E:\\saqib_work1\\data\\miles\\malware_in_class'
folders = os.listdir(dir)

for folder in folders:
    if os.path.isdir(os.path.join(dir, folder)):
        print("\n", folder)
        print('-----------------------------------------------')
        check_all_files(dir, folder, featureSize)
