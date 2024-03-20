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

script_directory = os.path.dirname(os.path.abspath(__file__))


#%% KODE TRAINING

# Load the data from the Excel file

data = pd.read_excel(script_directory + '/output_labelling' + '/dataset_satelit_latihan_1.xlsx')

# Split the data into features (spectral bands) and target label
X = data[['B2', 'B4']]  # Features: Spectral bands B2, B3, B4

y = data['jenis_lahan']        # Target label: Land cover types

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
results_df.to_excel(script_directory + '/prediction_results.xlsx', index=False)
