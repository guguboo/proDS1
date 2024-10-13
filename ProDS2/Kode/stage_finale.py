#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 08:36:55 2024

@author: MSI
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:39:38 2024

@author: MSI
"""

# import dataset

import rasterio
import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)
import addFeature as af
#%% TAHAP PERTAMA

fitur_terpilih = np.loadtxt(script_dir + '/selected_features.txt', dtype=str)

with open(script_dir + '/filename.txt', 'r') as file:
    content = file.read().strip()
    
features_stage_1 = ["EVI", "NDVI"]

labeled = parent_dir + "/Labeled/labeling_by_pixel_"
filename = content
df = pd.read_excel(labeled+filename)

X = df.copy()
y = df['land_cover']        

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

group_1 = ["agriculture", "forest"]
group_2 = ["land_without_scrub", "grassland", "river","crop", "road_n_railway"]
group_3 = ["tank", "settlement"]

land_cover = y_train
grouped_land_cover = []

for lc in land_cover:
    if lc in group_1:
        grouped_land_cover.append("group1")
    elif lc in group_2:
        grouped_land_cover.append("group2")
    else:
        grouped_land_cover.append("group3")



y_train_1 = grouped_land_cover

rfc_1 = RandomForestClassifier(
    bootstrap=True,
    max_depth=40,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=300,
    random_state=42  # To ensure reproducibility
)

rfc_1.fit(X_train[features_stage_1], grouped_land_cover)
hasil_1 = rfc_1.predict(X_test[features_stage_1])

print(len(hasil_1)) 

print(sum(hasil_1=="group1"))
df_g1 = X_test.iloc[hasil_1=="group1"]
df_g2 = X_test.iloc[hasil_1=="group2"]
df_g3 = X_test.iloc[hasil_1=="group3"]


print(df_g1['land_cover'].value_counts(),end="\n\n")
print(df_g2['land_cover'].value_counts(),end="\n\n")
print(df_g3['land_cover'].value_counts(),end="\n\n")

#%% TAHAP 2 GROUP 1

features_stage_2 = ['B11', 'B12']
X = df[df['land_cover'].isin(group_1)]
y = X['land_cover']

rfc_group1_1 = RandomForestClassifier(
    bootstrap=True,
    max_depth=40,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=300,
    random_state=42  # To ensure reproducibility
)
rfc_group1_1.fit(X[features_stage_2], y)

hasil_group1_1 = rfc_group1_1.predict(df_g1[features_stage_2])
res_group_1 = df_g1.copy()
res_group_1['prediction'] = hasil_group1_1

res_group_1

#%% TAHAP 2 GROUP 3

features_stage_3 = ['B11', 'B12']
X = df[df['land_cover'].isin(group_3)]
y = X['land_cover']

rfc_group3_1 = RandomForestClassifier(
    bootstrap=True,
    max_depth=40,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=300,
    random_state=42  # To ensure reproducibility
)

rfc_group3_1.fit(X[features_stage_3], y)

hasil_group3_1 = rfc_group3_1.predict(df_g3[features_stage_3])
res_group_3 = df_g3.copy()
res_group_3['prediction'] = hasil_group3_1

res_group_3


#%% TAHAP 2 GROUP 2
features_stage_2 = ['B11', 'B12', 'B8', 'B6']
X = df[df['land_cover'].isin(group_2)]
y = X['land_cover']

rfc_group2_1 = RandomForestClassifier(
    bootstrap=True,
    max_depth=40,
    max_features='log2',
    min_samples_leaf=2,
    min_samples_split=5,
    n_estimators=300,
    random_state=42  # To ensure reproducibility
)

rfc_group2_1.fit(X[features_stage_2], y)

hasil_group2_1 = rfc_group2_1.predict(df_g2[features_stage_2])
res_group_2 = df_g2.copy()
res_group_2['prediction'] = hasil_group2_1

res_group_2

#%% concat

res_akhir = pd.concat([res_group_1, res_group_2, res_group_3], axis=0)
res_akhir.head()

results_df = pd.DataFrame({'Actual': res_akhir['land_cover'], 'Predicted': res_akhir['prediction']})
results_df.head()
# UBAHHH FILENAME DI SINII --------------------------------------------------------------------------------
out_filename = "klasifikasi_bertahap_1.xlsx"
results_df.to_excel(parent_dir + "/Prediction/" + out_filename, index=False)


