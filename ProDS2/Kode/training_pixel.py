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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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
# UBAHHH FILENAME DI SINII ----------------------------------------------------------------------------------
filename = content

try:
    data = pd.read_excel(labeled+filename)
except:
    data = pd.read_excel(parent_dir + "/Labeled/cleaned_outliers/labeling_by_pixel_" + filename)
X = data[fitur_terpilih] 
y = data['land_cover']        

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 

#%% TRAINING cleaned_outliers

fitur_terpilih = np.loadtxt(script_dir + '/selected_features.txt', dtype=str)
# print(fitur_terpilih)
with open(script_dir + '/filename.txt', 'r') as file:
    content = file.read().strip()
labeled = parent_dir + "/Labeled/cleaned_outliers/labeling_by_pixel_"
# UBAHHH FILENAME DI SINII ----------------------------------------------------------------------------------
filename = content

data = pd.read_excel(labeled+filename)

X = data[fitur_terpilih] 
y = data['land_cover']        

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
#%% Initialize the Random Forest Classifier
'''
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

param_grid = {
    'n_estimators': [100, 300, 500],        # Number of trees
    'max_depth': [20, 40, None],              # Depth of each tree
    'min_samples_split': [2, 5, 10],                  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],                    # Minimum samples required at each leaf node
    'max_features': ['auto', 'sqrt', 'log2'],         # Number of features to consider for the best split
    'bootstrap': [True, False]                        # Whether bootstrap samples are used to build trees
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the model to find the best parameters
grid_search.fit(X, y)

# Print the best parameters
print(f"Best parameters: {grid_search.best_params_}")
'''

#%%
rf_optimized = RandomForestClassifier(
    bootstrap=True,
    max_depth=40,
    max_features='log2',
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=500,
    random_state=42  # To ensure reproducibility
)


# Train the classifier on the training data
rf_optimized.fit(X_train, y_train)

# Predict land cover types for the test data
y_pred = rf_optimized.predict(X_test)

# Create a DataFrame to store predicted and actual labels
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

# Export the results DataFrame to an Excel file

# UBAHHH FILENAME DI SINII --------------------------------------------------------------------------------
out_filename = content
results_df.to_excel(parent_dir + "/Prediction/" + out_filename, index=False)

def predict_real_data(filename, dta):
    fitur_terpilih = np.loadtxt(script_dir + '/selected_features.txt', dtype=str)
    global rf_optimized
    rfc = rf_optimized
    

    if int(filename.split(".")[0][-1]) > 5:    
        labeled = parent_dir + "/Labeled/cleaned_outliers/labeling_by_pixel_"
    else:
        labeled = parent_dir + "/Labeled/labeling_by_pixel_"
    
    training = pd.read_excel(labeled + filename)
    predict_data = pd.read_excel(parent_dir + '/Data/(to predict)/' + dta)
    
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
