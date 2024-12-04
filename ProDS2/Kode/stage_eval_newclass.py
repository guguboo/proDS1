#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:07:16 2024

@author: MSI
"""

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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)
import addFeature as af

#%%

def evaluation(truth, pred):
    truth_array = truth
    prediction_array = pred

    cm = confusion_matrix(truth_array, prediction_array)
    classes = sorted(list(set(truth_array)))  # ambil nama kelas

    # confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # hitung weighted
    curr_accuracy = accuracy_score(truth_array, prediction_array)
    curr_precision_w = precision_score(truth_array, prediction_array, average='weighted')
    curr_recall_w = recall_score(truth_array, prediction_array, average='weighted')
    curr_f1_w = f1_score(truth_array, prediction_array, average='weighted')
    
    curr_precision = precision_score(truth_array, prediction_array, average=None, labels=classes)
    curr_recall = recall_score(truth_array, prediction_array, average=None, labels=classes)
    curr_f1 = f1_score(truth_array, prediction_array, average=None, labels=classes)
    

    # print eval weighted (komen aja kalau gamau tiap kelas diprint
    print(f"Accuracy: {curr_accuracy:.4f}")
    print(f"Precision (weighted): {curr_precision_w:.4f}")
    print(f"Recall (weighted): {curr_recall_w:.4f}")
    print(f"F1 Score (weighted): {curr_f1_w:.4f}\n")
    
    for i, cls in enumerate(classes):
        print(f"  Class {cls}:")
        print(f"    Precision: {curr_precision[i] * 100:.2f}%")
        print(f"    Recall: {curr_recall[i] * 100:.2f}%")
        print(f"    F1 Score: {curr_f1[i] * 100:.2f}%")
    print("\n")
#%% TAHAP PERTAMA

fitur_terpilih = np.loadtxt(script_dir + '/selected_features.txt', dtype=str)

with open(script_dir + '/filename.txt', 'r') as file:
    content = file.read().strip()
    
features_stage_1 = ["B1", "B5", "B11", "B12"]

labeled = parent_dir + "/Labeled/labeling_by_pixel_"
filename = content
df = pd.read_excel(labeled+filename)

X = df.copy()
y = df['land_cover']        

# Split the data into training and testing sets

group_1 = ["agriculture", "forest"]
group_2 = ["land_without_scrub", "grassland", "tank_road_river","crop"]
group_3 = ["settlement"]

land_cover = y
grouped_land_cover = []

for lc in land_cover:
    if lc in group_1:
        grouped_land_cover.append("group1")
    elif lc in group_2:
        grouped_land_cover.append("group2")
    else:
        grouped_land_cover.append("group3")

y = grouped_land_cover

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#%%
'''
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

param_grid = {
    'n_estimators': [300],        # Number of trees
    'max_depth': [20, 40, None],              # Depth of each tree
    'min_samples_split': [5, 15],                  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],                    # Minimum samples required at each leaf node
    'max_features': ['auto', 'sqrt', 'log2'],         # Number of features to consider for the best split
}

grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid,
                           cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the model to find the best parameters
grid_search.fit(X_train[features_stage_1], y_train)

# Print the best parameters
print(f"Best parameters: {grid_search.best_params_}")
'''
#%%
rfc_1 = RandomForestClassifier(
    bootstrap=True,
    max_depth=40,
    max_features = 'sqrt',
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=300,
    random_state=42  # To ensure reproducibility
)

rfc_1.fit(X_train[features_stage_1], y_train)
hasil_1 = rfc_1.predict(X_test[features_stage_1])

print(len(hasil_1)) 

evaluation(y_test, hasil_1)

df_g1 = X_test.iloc[hasil_1=="group1"]
df_g2 = X_test.iloc[hasil_1=="group2"]
df_g3 = X_test.iloc[hasil_1=="group3"]

print(df_g1)


# print(df_g1['land_cover'].value_counts(),end="\n\n")
# print(df_g2['land_cover'].value_counts(),end="\n\n")
# print(df_g3['land_cover'].value_counts(),end="\n\n")

#%% TAHAP 2 GROUP 1

features_stage_2 = ['B3', 'B4', 'B5', 'B8', 'B11']
X = df[df['land_cover'].isin(group_1)]
y = X['land_cover']

rfc_group1_1 = RandomForestClassifier(
    bootstrap=True,
    max_depth=40,
    max_features = 'sqrt',
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=300,
    random_state=42  # To ensure reproducibility
)
rfc_group1_1.fit(X[features_stage_2], y)

hasil_group1_1 = rfc_group1_1.predict(df_g1[features_stage_2])

evaluation(df_g1['land_cover'], hasil_group1_1)
res_group_1 = df_g1.copy()
res_group_1['prediction'] = hasil_group1_1

res_group_1

#%% TAHAP 2 GROUP 3

features_stage_3 = ['B1', 'B2', 'B4', 'B5', 'B11', 'B12']
X = df[df['land_cover'].isin(group_3)]
y = X['land_cover']

rfc_group3_1 = RandomForestClassifier(
    bootstrap=True,
    max_depth=40,
    max_features = 'sqrt',
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=300,
    random_state=42  # To ensure reproducibility
)

rfc_group3_1.fit(X[features_stage_3], y)

hasil_group3_1 = rfc_group3_1.predict(df_g3[features_stage_3])

evaluation(df_g3['land_cover'], hasil_group3_1)

res_group_3 = df_g3.copy()
res_group_3['prediction'] = hasil_group3_1



res_group_3


#%% TAHAP 2 GROUP 2
features_stage_2 = ['B5', 'B7', 'B8', 'B12']
X = df[df['land_cover'].isin(group_2)]
y = X['land_cover']

rfc_group2_1 = RandomForestClassifier(
    bootstrap=True,
    max_depth=40,
    max_features = 'sqrt',
    min_samples_leaf=1,
    min_samples_split=5,
    n_estimators=300,
    random_state=42  # To ensure reproducibility
)

rfc_group2_1.fit(X[features_stage_2], y)

hasil_group2_1 = rfc_group2_1.predict(df_g2[features_stage_2])

evaluation(df_g2['land_cover'], hasil_group2_1)

res_group_2 = df_g2.copy()
res_group_2['prediction'] = hasil_group2_1

res_group_2

#%% concat

res_akhir = pd.concat([res_group_1, res_group_2, res_group_3], axis=0)
res_akhir.head()

results_df = pd.DataFrame({'Actual': res_akhir['land_cover'], 'Predicted': res_akhir['prediction']})
results_df.head()
# UBAHHH FILENAME DI SINII --------------------------------------------------------------------------------
out_filename = "klasifikasi_bertahap_4.xlsx"
results_df.to_excel(parent_dir + "/Prediction/" + out_filename, index=False)


