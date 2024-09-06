# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:11:14 2024

@author: asus FX506HC
"""

#%%

import os
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns

script_directory = os.path.dirname(os.path.abspath(__file__))
#%% read excel dan masukan ke variabel (CONTOH)

df = pd.read_excel(script_directory + "/prediction_results.xlsx")

prediction_array = df['Predicted']
true_array = df['Actual']


#%% function buat matrix (andrea version)

def evaluation_function(prediction_array, truth_array, bands):
    
    global f1_air_max, f1_bangunan_max, f1_area_hijau_max
    global band_air, band_bangunan, band_area_hijau
    # Compute confusion matrix
    cm = confusion_matrix(truth_array, prediction_array)
    classes = set(truth_array.unique())
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Compute metrics
    accuracy = accuracy_score(truth_array, prediction_array)
    precision = precision_score(truth_array, prediction_array, average=None)
    recall = recall_score(truth_array, prediction_array, average=None)
    f1 = f1_score(truth_array, prediction_array, average=None)
    
    # Print metrics
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision")
    for i, cls in enumerate(classes):
        print("Class", cls, ":", "{:.2f}%".format(precision[i] * 100))
    print("Recall")
    for i, cls in enumerate(classes):  
        print("Class", cls, ":", "{:.2f}%".format(recall[i] * 100))
    print("F1 score")
    for i, cls in enumerate(classes):
        print("Class", cls, ":", "{:.2f}%".format(f1[i] * 100))

    f1_weighted = f1_score(truth_array, prediction_array, average='weighted')
    print("F1 Score (weighted): {:.2f}%".format(f1_weighted * 100))
    print()

evaluation_function(prediction_array, true_array, [""])
    
#%% reset best variables

band_air = []
band_bangunan = []
band_area_hijau = []
band_weighted = []

f1_air_max = 0
f1_bangunan_max = 0
f1_area_hijau_max = 0
best_weighted = 0

#%% function buat matrix (VICO VERSION)

def evaluation_function(prediction_array, truth_array, bands):
    
    global f1_air_max, f1_bangunan_max, f1_area_hijau_max, best_weighted
    global band_air, band_bangunan, band_area_hijau, band_weighted

    classes = set(truth_array.unique())
    
    f1 = f1_score(truth_array, prediction_array, average=None)
    
    for i, cls in enumerate(classes):
        if(cls == "air"):
            if(f1[i] > f1_air_max):
                f1_air_max = f1[i]
                band_air = bands
        elif(cls == "bangunan"):
            if(f1[i] > f1_bangunan_max):
                f1_bangunan_max = f1[i]
                band_bangunan = bands
        else:
            if(f1[i] > f1_area_hijau_max):
                f1_area_hijau_max = f1[i]
                band_area_hijau = bands
        
    f1_weighted = f1_score(truth_array, prediction_array, average='weighted')
    if(f1_weighted > best_weighted):
        best_weighted = f1_weighted
        band_weighted = bands
        
#%% Check untuk setiap kombinasi

X = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12']
    
for i in range (1, 11):
    comb = combinations(X, i)
     
    for i in list(comb): 
        selected = list(i)
        nama_file = "prediction_result"
        
        for b in selected:
            nama_file += "_" + b
            
        df = pd.read_excel(script_directory + "/prediction_result/combination(all_bands)/" + nama_file + ".xlsx")
    
        prediction_array = df['Predicted']
        true_array = df['Actual']
            
        evaluation_function(prediction_array, true_array, selected)
        
    print(f"done for combination {i}")
    
print(f"{band_air} is the best for air, with f1 score: {f1_air_max * 100:.2f}%")
print(f"{band_bangunan} is the best for bangunan, with f1 score: {f1_bangunan_max * 100:.2f}%")
print(f"{band_area_hijau} is the best for area_hijau, with f1 score: {f1_area_hijau_max * 100:.2f}%")
print(f"{band_weighted} is the best for overall, with f1 score: {best_weighted * 100:.2f}%")


