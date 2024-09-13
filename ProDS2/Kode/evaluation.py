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

df1 = pd.read_excel("D:/Andrea/UNPAR/Sem 6/prods/proDS1/prediction_results.xlsx")
df2 = pd.read_excel("D:/Andrea/UNPAR/Sem 6/prods/proDS1/prediction_presentasi.xlsx")
df3 = pd.read_excel("D:/Andrea/UNPAR/Sem 6/prods/proDS1/prediksi_grid.xlsx")
df4 = pd.read_excel("D:/Andrea/UNPAR/Sem 6/prods/proDS1/prediksi_presentasi.xlsx")

#%% function buat matrix (andrea version)

def evaluation_function(results, bands):
    
    global f1_air_max, f1_bangunan_max, f1_area_hijau_max
    global band_air, band_bangunan, band_area_hijau

    # variabel buat nyimpen nilai eval
    accuracy = precision_w = recall_w = f1_w = 0.0
    best_accuracy = best_precision_w = best_recall_w = best_f1_w = -1

    precision = recall = f1 = None
    best_precision = best_recall = best_f1 = None

    for idx, df in enumerate(results):  # eval tiap dataframe
        truth_array = df['Actual']  
        prediction_array = df['Predicted']  

        cm = confusion_matrix(truth_array, prediction_array)
        classes = sorted(list(set(truth_array.unique())))  # ambil nama kelas

        # confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for DataFrame {idx}')
        plt.show()

        # hitung weighted
        curr_accuracy = accuracy_score(truth_array, prediction_array)
        curr_precision_w = precision_score(truth_array, prediction_array, average='weighted')
        curr_recall_w = recall_score(truth_array, prediction_array, average='weighted')
        curr_f1_w = f1_score(truth_array, prediction_array, average='weighted')
        
        # eval metrics per class
        curr_precision = precision_score(truth_array, prediction_array, average=None, labels=classes)
        curr_recall = recall_score(truth_array, prediction_array, average=None, labels=classes)
        curr_f1 = f1_score(truth_array, prediction_array, average=None, labels=classes)

        # buat nyimpen nilai eval per kelas
        if precision is None:
            num_classes = len(classes)
            precision = [-1] * num_classes
            recall = [-1] * num_classes
            f1 = [-1] * num_classes
            best_precision = [-1] * num_classes
            best_recall = [-1] * num_classes
            best_f1 = [-1] * num_classes

        # print eval weighted (komen aja kalau gamau tiap kelas diprint)
        print(f"DataFrame {idx}")
        print(f"Accuracy: {curr_accuracy:.4f}")
        print(f"Precision (weighted): {curr_precision_w:.4f}")
        print(f"Recall (weighted): {curr_recall_w:.4f}")
        print(f"F1 Score (weighted): {curr_f1_w:.4f}\n")

        # print eval tiap kelas
        for i, cls in enumerate(classes):
            print(f"  Class {cls}:")
            print(f"    Precision: {curr_precision[i] * 100:.2f}%")
            print(f"    Recall: {curr_recall[i] * 100:.2f}%")
            print(f"    F1 Score: {curr_f1[i] * 100:.2f}%")
        print("\n")
        
        # simpen nilai eval yg paling bagus
        # weighted
        if curr_accuracy > accuracy:
            accuracy = curr_accuracy
            best_accuracy = idx
            
        if curr_precision_w > precision_w:
            precision_w = curr_precision_w
            best_precision_w = idx
            
        if curr_recall_w > recall_w:
            recall_w = curr_recall_w
            best_recall_w = idx
            
        if curr_f1_w > f1_w:
            f1_w = curr_f1_w
            best_f1_w = idx

        # per kelas
        for i in range(len(classes)): 
            if curr_precision[i] > precision[i]:
                precision[i] = curr_precision[i]
                best_precision[i] = idx
                
            if curr_recall[i] > recall[i]:
                recall[i] = curr_recall[i]
                best_recall[i] = idx  
                
            if curr_f1[i] > f1[i]:
                f1[i] = curr_f1[i]
                best_f1[i] = idx  

    # print weighted metrics yg paling bagus dari semua df yg dites
    print(f"Best Accuracy: {best_accuracy} ({accuracy * 100:.2f}%)")
    print(f"Best Precision (weighted): {best_precision_w} ({precision_w * 100:.2f}%)")
    print(f"Best Recall (weighted): {best_recall_w} ({recall_w * 100:.2f}%)")
    print(f"Best F1 score (weighted): {best_f1_w} ({f1_w * 100:.2f}%)")
    
    # print metrics per kelas yg paling bagus dan ada di df yg mana
    print("\nClass-specific metrics:")
    for i, cls in enumerate(classes):
        print(f"Class {cls}:")
        print(f"  Best Precision: {precision[i] * 100:.2f}% (DataFrame {best_precision[i]})")
        print(f"  Best Recall: {recall[i] * 100:.2f}% (DataFrame {best_recall[i]})")
        print(f"  Best F1 Score: {f1[i] * 100:.2f}% (DataFrame {best_f1[i]})\n")

# run
dfs = [df1, df2, df3, df4]
evaluation_function(dfs, [""])

   

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


