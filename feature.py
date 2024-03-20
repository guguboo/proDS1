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
import statsmodels.api as sm
from statsmodels.formula.api import ols


# %% dataset iris

df = pd.read_excel("/Users/kevinchristian/Downloads/dataset_satelit_latihan_1.xlsx")

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
    anova_results = anova_results.sort_values(by='p_value', ascending=True)
    
    # print 5 fitur dengan p-value terendah
    print(anova_results.head())
    print()
    
    
    alpha = 0.05
    
    # memilih fitur berdasarkan p-value
    selected_features = features.columns[anova_results['p_value'] < alpha]
    
    print("Selected features:", selected_features)
    print()
    

anova_test(features, target)

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
k = 2

selected_features_k_best = select_k_best(features, target, k)
print(f"Selected {k} best features based on SelectKBest: {selected_features_k_best}")

# %% pearson correlation

df_cor = features.corr()
plt.figure(figsize=(10,10))
sns.heatmap(df_cor, annot=True)
plt.show()

