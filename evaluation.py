# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:11:14 2024

@author: asus FX506HC
"""

#%%
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

#%% contoh arraynya
prediction_array = [1, 3, 2, 2, 1, 1]
true_array = [1, 1, 3, 2, 2, 1]

#%% function buat matrix
def evaluation_function(prediction_array, truth_array):
    # Compute confusion matrix
    cm = confusion_matrix(truth_array, prediction_array)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=set(truth_array), yticklabels=set(truth_array))
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
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    f1 = f1_score(truth_array, prediction_array, average='weighted')
    print("F1 Score (weighted):", f1)

evaluation_function(prediction_array, true_array)
#%%