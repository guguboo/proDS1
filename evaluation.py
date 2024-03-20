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
import seaborn as sns

<<<<<<< Updated upstream
#%% read excel dan masukan ke variabel
df = pd.read_excel("D:/Andrea/UNPAR/Sem 6/prods/proDS1/proDS1/prediction_results.xlsx")
=======
#%% contoh arraynya

script_directory = os.path.dirname(os.path.abspath(__file__))
df = pd.read_excel(script_directory + "/prediction_results.xlsx")
>>>>>>> Stashed changes
prediction_array = df['Predicted']
true_array = df['Actual']
classes = set(true_array.unique())

#%% function buat matrix
def evaluation_function(prediction_array, truth_array):
    # Compute confusion matrix
    cm = confusion_matrix(truth_array, prediction_array)
    
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

evaluation_function(prediction_array, true_array)
#%%