#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:06:16 2024

@author: kevinchristian
"""
# %% import libraries

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
import random
from sklearn.feature_selection import SelectKBest
import numpy as np
from scipy import stats


script_directory = os.path.dirname(os.path.abspath(__file__))
# %% dataset iris

df = pd.read_excel(script_directory + "/output_labelling/dataset_grid_statistik_2x2.xlsx")

# %% visualisasi

features = df.iloc[:, :-1]  # semua kolom kecuali kolom target
target = df.iloc[:, -1] 

num_categories = len(target.unique())

# get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
# my_palette_1 = get_colors(num_categories) 

my_palette_1 = ["black","green","blue"]

for feature in features.columns:
    with plt.rc_context(rc = {'figure.dpi': 150, 'axes.labelsize': 9, 
                              'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                              'legend.fontsize': 8.5, 'legend.title_fontsize': 9}):
    
        fig_2, ax_2 = plt.subplots(2, 2, figsize = (15, 10)) 
        
        sns.kdeplot(ax = ax_2[0, 0], x = df[feature], linewidth = 1.5,
                    hue = df[target.name], common_norm = True,
                    fill = True, alpha = 0.4, palette = my_palette_1)
        
        sns.stripplot(ax = ax_2[0, 1], x = df[target.name], s = 2,
                      y = df[feature], palette = my_palette_1, alpha = 1)
        
        ax_2[1, 0].set_visible(False)
        ax_2[1, 1].set_visible(False)
        
        plt.tight_layout(pad = 1.5)
        plt.show()

# %% anova test

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

    
    print("Selected features:", selected_features)
    print()
    
    return selected_features


selected_features = anova_test(features, target)

# model = ols('B4 ~ jenis_lahan',data=df).fit()
# print(sm.stats.anova_lm(model).round(5))


# %% pilih fitur terbaik

def select_k_best(features, target, k):
    # Membuat objek SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Fit dan transformasi fitur
    features_selected = selector.fit_transform(features, target)
    
    # Mendapatkan indeks fitur terpilih
    selected_indices = selector.get_support(indices=True)
    
    # Mendapatkan nama fitur terpilih
    selected_features = features.columns[selected_indices]
    
    return selected_features

# Jumlah fitur yang ingin dipilih
k = 10

selected_features_k_best = select_k_best(features, target, k)
print(f"Selected {k} best features based on SelectKBest: {selected_features_k_best}")

# %% pearson correlation

df_cor = features.corr()
plt.figure(figsize=(10,10))
sns.heatmap(df_cor, annot=True)
plt.show()

