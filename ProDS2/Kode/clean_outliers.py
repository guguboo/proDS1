#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 11:44:47 2024

@author: MSI
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

#%%

fitur_terpilih = np.loadtxt(script_dir + '/selected_features.txt', dtype=str)
# print(fitur_terpilih)
with open(script_dir + '/filename.txt', 'r') as file:
    content = file.read().strip()
labeled = parent_dir + "/Labeled/labeling_by_pixel_"
# UBAHHH FILENAME DI SINII ----------------------------------------------------------------------------------
filename = content

data = pd.read_excel(labeled+filename)
data.shape
# data.columns

#%% clean outliers

def remove_outliers(df, column_list, group_column='land_cover'):
    df_cleaned = pd.DataFrame()  # Empty DataFrame to store cleaned data

    for name, group in df.groupby(group_column):
        # Copy group to modify in place
        group_cleaned = group.copy()
        
        for col in column_list:
            Q1 = group[col].quantile(0.25)
            Q3 = group[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define lower and upper bounds for the IQR method
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Remove outliers
            group_cleaned = group_cleaned[(group_cleaned[col] >= lower_bound) & (group_cleaned[col] <= upper_bound)]
        
        # Append the cleaned group to the cleaned DataFrame
        df_cleaned = pd.concat([df_cleaned, group_cleaned], axis=0)
    
    # Reset index of the cleaned DataFrame
    df_cleaned = df_cleaned.reset_index(drop=True)
    return df_cleaned

# Columns to clean for outliers (excluding 'land_cover')
columns_to_check = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B11', 'B12', 'NDVI', 'EVI', 'NDWI']

# Clean the DataFrame
df_cleaned = remove_outliers(data, columns_to_check)
df_cleaned.shape

df_cleaned.to_excel(parent_dir + '/Labeled/cleaned_outliers/labeling_by_pixel_' + filename, index=False)
#%% viz

features = df_cleaned.iloc[:, :-1]  # semua kolom kecuali kolom target
target = df_cleaned.iloc[:, -1] 

ndvi_evi = features[['NDVI', 'EVI', 'NDWI']]

other_features = features.drop(columns=['NDVI', 'EVI', 'NDWI'])

scaler = MinMaxScaler()
other_features_scaled = pd.DataFrame(scaler.fit_transform(other_features), columns=other_features.columns)

features_scaled = pd.concat([other_features_scaled, ndvi_evi], axis=1)
label_order = ['agriculture', 'forest', 'crop', 'road_n_railway', 'river','land_without_scrub', 'grassland', 'tank', 'settlement']

for feature in features_scaled.columns:
    plt.figure(figsize=(35, 20))
    
    sns.boxplot(x=target, y=features_scaled[feature], color='skyblue', order=label_order)
    
    if feature in ['NDVI', 'EVI', 'NDWI']:
        plt.ylim(-1, 1)
    elif feature in ['B11', 'B12']:
        plt.ylim(0, 1)
    elif feature in ['B6', 'B7', 'B8', 'B1']:        
        plt.ylim(0, 1)
    else:
        plt.ylim(0, 1)
    
    plt.title(f'Box Plot {feature}', fontsize=30, weight='bold')
    plt.xticks(fontsize=25, weight='bold')
    plt.yticks(fontsize=30, weight='bold')
    
    plt.xlabel('')
    
    plt.show()

filtered_features_scaled = features_scaled.drop(columns=['NDVI', 'EVI', 'NDWI'])
