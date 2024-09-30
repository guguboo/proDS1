#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 12:42:37 2024

@author: MSI
"""

import os
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from scipy import stats
import pandas as pd
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

#%% UJI ANOVA

labeled = parent_dir + "/Labeled/labeling_by_pixel_"
filename = "ProDS2.xlsx"
df = pd.read_excel(labeled+filename)

features = df.iloc[:, :-1]  # semua kolom kecuali kolom target
target = df.iloc[:, -1] 

def anova_test(features, target):
    # hitung F-value, p-value untuk setiap fitur
    f_values, p_values = f_classif(features, target)
    
    # buat dataframe untuk menyimpan hasil perhitungan
    anova_results = pd.DataFrame({'feature': features.columns, 'f_value': f_values, 'p_value': p_values})
    
    # sort berdasarkan p-value
    anova_results = anova_results.sort_values(by='f_value', ascending=False)
    
    # print 5 fitur dengan p-value terendah
    print(anova_results.head())
    print()
    
    alpha = 0.05
    
    # Mencetak hasil perbandingan antara nilai F dan nilai kritis F
    for i, row in anova_results.iterrows():
        f_crit = stats.f.ppf(1 - alpha, len(features.columns) - 1, len(features) - len(features.columns))
        if row['f_value'] > f_crit:
            print(f"Feature '{row['feature']}': F-value ({row['f_value']:.4f}) > F-critical ({f_crit:.4f}), significant.")
        else:
            print(f"Feature '{row['feature']}': F-value ({row['f_value']:.4f}) <= F-critical ({f_crit:.4f}), not significant.")
    
    print() 
   
    # Membuat larik boolean yang menunjukkan apakah nilai F lebih besar dari nilai kritis F
    significant_f = anova_results['f_value'] > f_crit

    # Memilih fitur yang memenuhi kriteria F-value > F-critical
    selected_features = features.columns[significant_f]

    
    return selected_features


selected_features = anova_test(features, target)

def select_k_best(features, target, k):
    # Membuat objek SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Fit dan transformasi fitur
    selector.fit_transform(features, target)
    
    # Mendapatkan indeks fitur terpilih
    selected_indices = selector.get_support(indices=True)
    
    # Mendapatkan nama fitur terpilih
    selected_features = features.columns[selected_indices]
    
    return selected_features

# Jumlah fitur yang ingin dipilih
k = 6

selected_features_k_best = select_k_best(features, target, k)
print(f"Selected {k} best features based on SelectKBest: {selected_features_k_best}")

array = np.array(selected_features_k_best)

# Write the array to a txt file
np.savetxt(script_dir + '/selected_features.txt', array, fmt='%s')
