#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:07:49 2024

@author: MSI
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

#%% TRAINING


# Read the array from the txt file
fitur_terpilih = np.loadtxt(script_dir + '/selected_features.txt', dtype=str)
# print(fitur_terpilih)
with open(script_dir + '/filename.txt', 'r') as file:
    content = file.read().strip()
labeled = parent_dir + "/Labeled/labeling_by_pixel_"
# UBAHHH FILENAME DI SINII --------------------------------------------------------------------------------
filename = content

data = pd.read_excel(labeled+filename)

X = data[fitur_terpilih] 
y = data['land_cover']        

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict land cover types for the test data
y_pred = rf_classifier.predict(X_test)

# Create a DataFrame to store predicted and actual labels
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

# Export the results DataFrame to an Excel file

# UBAHHH FILENAME DI SINII --------------------------------------------------------------------------------
out_filename = content
results_df.to_excel(parent_dir + "/Prediction/" + out_filename, index=False)

def predict_real_data(filename):
    fitur_terpilih = np.loadtxt(script_dir + '/selected_features.txt', dtype=str)
    rfc = RandomForestClassifier(n_estimators=50, random_state=42)
    
    labeled = parent_dir + "/Labeled/labeling_by_pixel_"
    training = pd.read_excel(labeled + filename)
    predict_data = pd.read_excel(parent_dir + '/Data/(to predict)/' + filename)
    
    x = training[fitur_terpilih] 
    y = training['land_cover']  

    rfc.fit(x, y)
    predict_features = predict_data[fitur_terpilih]
    predict_features = predict_features.replace([np.inf, -np.inf], np.nan)
    predict_features = predict_features.fillna(predict_features.mean())
    
    y_pred = rfc.predict(predict_features)
    nama_file = filename
    results_df = pd.DataFrame({'x': predict_data['x'], 'y': predict_data['y'], 'land_cover': y_pred})
    results_df.to_excel(parent_dir + '/Result/' + nama_file, index=False)
    
    return results_df
