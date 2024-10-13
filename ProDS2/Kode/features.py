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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

#%% UJI ANOVA
with open(script_dir + '/filename.txt', 'r') as file:
    content = file.read().strip()

labeled = parent_dir + "/Labeled/labeling_by_pixel_"
filename = content
df = pd.read_excel(labeled+filename)
cols = ["land_without_scrub", "grassland", "river", "crop", "road_n_railway"]

df = df[df['land_cover'].isin(cols)]
print(df['land_cover'].value_counts())
features = df.iloc[:, :-1]  # semua kolom kecuali kolom target
target = df['land_cover'] 

 #%%normalisasi
ndvi_evi = features[['NDVI', 'EVI']]

other_features = features.drop(columns=['NDVI', 'EVI'])

scaler = MinMaxScaler()
other_features_scaled = pd.DataFrame(scaler.fit_transform(other_features), columns=other_features.columns)

features_scaled = pd.concat([other_features_scaled, ndvi_evi], axis=1)

label_order = cols
# label_order = ['agriculture', 'forest', 'crop', 'road_n_railway', 'river','land_without_scrub', 'grassland', 'tank', 'settlement']

for feature in features_scaled.columns:
    plt.figure(figsize=(35, 20))
    
    sns.boxplot(x=target, y=features_scaled[feature], color='skyblue', order=label_order)
    
    if feature in ['NDVI', 'EVI', 'NDWI']:
        plt.ylim(-0.3, 1)
    elif feature in ['B6', 'B7', 'B8']:
        plt.ylim(0, 0.6)
    else:
        plt.ylim(0, 0.6)
        
    
    plt.title(f'Box Plot {feature}', fontsize=30, weight='bold')
    plt.xticks(fontsize=20, weight='bold')
    plt.yticks(fontsize=20, weight='bold')
    
    plt.xlabel('')
    
    plt.show()

# filtered_features_scaled = features_scaled.drop(columns=['NDVI', 'EVI'])

# for label in target.unique():
#     plt.figure(figsize=(20, 15))
    
#     sns.boxplot(data=filtered_features_scaled[target == label], color='skyblue')
    
#     plt.ylim(0, 1)
    
#     plt.title(f'Box Plot for Label: {label}', fontsize=30, weight='bold')
    
#     plt.xticks(fontsize=30, weight='bold')
#     plt.yticks(fontsize=20, weight='bold')
        
#     plt.show()

#%% visualisasi

global_min = features.min().min()
global_max = features.max().max()

for feature in features.columns:
    plt.figure(figsize=(35, 20))
    
    sns.boxplot(x=target, y=df[feature], color='skyblue')
    
    if feature in ['NDVI', 'EVI']:
        plt.ylim(-1, 1)
    else:
        plt.ylim(global_min, global_max)
    
    plt.title(f'Box Plot {feature}', fontsize=30, weight='bold')
    plt.xticks(fontsize=20, weight='bold')
    plt.yticks(fontsize=20, weight='bold')
    
    plt.xlabel('')
    
    plt.show()
    
    
filtered_features = df[features.columns].drop(columns=['NDVI', 'EVI'])

for label in target.unique():
    plt.figure(figsize=(20, 15))
    
    sns.boxplot(data=filtered_features[target == label], color='skyblue')
    
    plt.ylim(global_min, global_max)
    
    plt.title(f'Box Plot for Label: {label}', fontsize=30, weight='bold')
    
    plt.xticks(fontsize=30, weight='bold')
    plt.yticks(fontsize=20, weight='bold')
        
    plt.show()

#%% uji
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
k = 8

selected_features_k_best = select_k_best(features, target, k)
print(f"Selected {k} best features based on SelectKBest: {selected_features_k_best}")

array = np.array(selected_features_k_best)

# Write the array to a txt file
np.savetxt(script_dir + '/selected_features.txt', array, fmt='%s')
