# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 11:58:23 2024

@author: hp
"""

#%% IMPORT IMPORTAN
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from itertools import combinations


script_directory = os.path.dirname(os.path.abspath(__file__))


#%% KODE TRAINING (MARK VERSION)

# Load the data from the Excel file

data = pd.read_excel(script_directory + '/output_labelling' + '/dataset_grid_metode_3.xlsx')

# Split the data into features (spectral bands) and target label


X = data[['B1', 'B2', 'B4', 'B5', 'B11', 'B12']]  # Features: Spectral bands B2, B3, B4
# X = data[['B4', 'B6', 'B7', 'B8', 'B11', 'B12']]  

y = data['jenis_lahan']        # Target label: Land cover types

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict land cover types for the test data
y_pred = rf_classifier.predict(X_test)

# Create a DataFrame to store predicted and actual labels
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

# Export the results DataFrame to an Excel file
results_df.to_excel(script_directory + '/prediction_results.xlsx', index=False)



#%% KODE TRAINING UNTUK EVALUASI(VICO VERSION)

def train(x, y, bands):
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    rf_classifier.fit(X_train, y_train)

    y_pred = rf_classifier.predict(X_test)

    results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

    nama_file = "prediction_result"
    
    for b in bands:
        nama_file += "_" + b
        
    results_df.to_excel(script_directory + '/prediction_result/combination(all_bands)/' + nama_file + '.xlsx', index=False)
#%% KODE TRAINING COBA 20m

# Load the data from the Excel file

data = pd.read_excel(script_directory + '/output_labelling' + '/dataset_grid_metode_2.xlsx')


data.groupby("jenis_lahan").count()


#data kalo baseline yg ini
X = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
y = data['jenis_lahan']
    
for i in range (1, 11):
    comb = combinations(X, i) 
     
    for i in list(comb): 
        selected = list(i)
        x = data[selected]
        train(x, y, selected)
        
    print(f"combination ke-{i} done...")
print("done creating prediction files...")


#%% KODE TRAINING UNTUK PREDIKSI(VICO VERSION)

def train_only(x_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    return rf_classifier

#%% PREDIKSI ONLY

def predict_only(rfc, data, x_coor, y_coor):
    y_pred = rfc.predict(data)
    results_df = pd.DataFrame({'x': x_coor, 'y': y_coor, 'jenis_lahan': y_pred})
    nama_file = "hasil_prediksi_grid_2"
    results_df.to_excel(script_directory + '/prediction_result/real_predict/' + nama_file + '.xlsx', index=False)

    return y_pred

#%% KODE TRAINING COBA 20m

training_data = pd.read_excel(script_directory + '/output_labelling' + '/dataset_grid_metode_2.xlsx')

#data kalo baseline yg ini
# x = ['B1', 'B2', 'B6', 'B8', 'B12']
x = ['B1', 'B2', 'B3', 'B4', 'B11', 'B12'] 

x_train = training_data[x]
y_train = training_data['jenis_lahan']
    
rfc = train_only(x_train, y_train)

#%% prediksi sesungguhnya

predict_data = pd.read_excel(script_directory + '/coba_remapping' + '/dataset_prediksi_grid.xlsx')
print(predict_data)


#%% prediksi
bands = ['B1', 'B2', 'B3', 'B4', 'B11', 'B12'] 

hasil = predict_only(rfc, predict_data[bands], predict_data['x'], predict_data['y'])

print(hasil)


#%% DUA KALI DUA GRID

# Load the data from the Excel file

data = pd.read_excel(script_directory + '/output_labelling' + '/dataset_grid_metode_3.xlsx')


data.groupby("jenis_lahan").count()


#data kalo baseline yg ini
X = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
y = data['jenis_lahan']
    
for i in range (1, 11):
    comb = combinations(X, i) 
     
    for i in list(comb): 
        selected = list(i)
        x = data[selected]
        train(x, y, selected)
        
    print(f"combination ke-{i} done...")
print("done creating prediction files...")


#%% KODE TRAINING UNTUK PREDIKSI(VICO VERSION)

def train_only(x_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    return rf_classifier

#%% PREDIKSI ONLY

def predict_only(rfc, data, x_coor, y_coor):
    y_pred = rfc.predict(data)
    results_df = pd.DataFrame({'x': x_coor, 'y': y_coor, 'jenis_lahan': y_pred})
    nama_file = "hasil_prediksi_grid_2x2"
    results_df.to_excel(script_directory + '/prediction_result/real_predict/' + nama_file + '.xlsx', index=False)

    return y_pred

#%% KODE TRAINING COBA 20m

training_data = pd.read_excel(script_directory + '/output_labelling' + '/dataset_grid_metode_3.xlsx')

#data kalo baseline yg ini
# x = ['B1', 'B2', 'B6', 'B8', 'B12']
x = ['B1', 'B2', 'B3', 'B4', 'B6', 'B12'] 

x_train = training_data[x]
y_train = training_data['jenis_lahan']
    
rfc = train_only(x_train, y_train)

#%% prediksi sesungguhnya

predict_data = pd.read_excel(script_directory + '/coba_remapping' + '/dataset_prediksi_grid_2x2.xlsx')
print(predict_data)


#%% prediksi
bands = ['B1', 'B2', 'B3', 'B4', 'B6', 'B12'] 

hasil = predict_only(rfc, predict_data[bands], predict_data['x'], predict_data['y'])

print(hasil)

