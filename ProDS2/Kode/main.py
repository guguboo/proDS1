#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 22:30:27 2024

@author: MSI
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:35:35 2024

@author: MSI
"""

#%% import dataset

import rasterio
import statistics

import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from shapely import Polygon
import geopandas as gpd
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, accuracy_score
# from itertools import combinations
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from scipy import stats

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
#%%

bands_src = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

band_path = parent_dir + '/Data/satelit/21082024/'

for i in range(1, 13):
    if i != 9 and i != 10:
        curr_band = rasterio.open(band_path + "B" + str(i) + ".jp2")
        bands_src[i] = curr_band

print("Successfully Read Bands Src")

#%% src to raster

bands_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(1, 13):
    if i != 9 and i != 10:
        bands_list[i] = bands_src[i].read(1)

print("Successfully Read Bands Raster")
#%% LABELLING
label_andrea = parent_dir + "/Labeling/andrea_"
n_andrea = 1
class_andrea = ["crop", "agriculture"]

label_kevin = parent_dir + "/Labeling/kevin_"
n_kevin = 1
class_kevin = ["grassland", "settlement", "road_n_railway"]

label_vico = parent_dir + "/Labeling/vico_"
n_vico = 0
class_vico = ["forest", "land_without_scrub", "land"]

label_mark = parent_dir + "/Labeling/mark_"
n_mark = 0
class_mark = ["forest", "land_without_scrub", "land"]

all_labels = [(label_andrea, n_andrea, class_andrea), 
              (label_kevin, n_kevin, class_kevin), 
              (label_vico, n_vico, class_vico), 
              (label_mark, n_mark, class_mark)]

src = bands_src[1]

bands_output = [[], [], [], [], [], [], [], [], [], [], [], [], []]
label_output = []

for label in all_labels:
    label_file = label[0]
    label_count = label[1]
    label_class = label[2]
    out_of_bound_count = 0
    for i in range(1, label_count + 1):
        print("Processing file " + label_file + str(i))
        multipoints_gdf = gpd.read_file(label_file + str(i) + ".geojson")
        multipoints_gdf = multipoints_gdf.to_crs(src.crs)
        
        for index, kategori in multipoints_gdf.iterrows():s
            multipoint_geometry = kategori['geometry']
            
            for point in multipoint_geometry.geoms:
                
                x, y = point.x, point.y
                x_raster, y_raster = src.index(x, y)
                try:
                    for i in range(1, 13):
                        if i != 9 and i != 10:
                            bands_output[i].append(bands_list[i][x_raster][y_raster])
                    label_output.append(label_class[index])
                except:
                    out_of_bound_count += 1
        print("Done, out of bound coordinates:", out_of_bound_count)

#%% output ke excel
output_filename = 'labelling_by_pixel_ProDS2'

out_df = pd.DataFrame({
    'B1': bands_output[1],
    'B2': bands_output[2],
    'B3': bands_output[3],
    'B4': bands_output[4],
    'B5': bands_output[5],
    'B6': bands_output[6],
    'B7': bands_output[7],
    'B8': bands_output[8],
    'B11': bands_output[11],
    'B12': bands_output[12],
    'land_cover': label_output
    })

out_df = out_df.drop_duplicates()

print(out_df.shape)

out_df.to_excel(parent_dir + '/Labeled/' + output_filename + ".xlsx", index=False)
#%% UJI ANOVA

features = out_df.iloc[:, :-1]  # semua kolom kecuali kolom target
target = out_df.iloc[:, -1] 

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

#%% TRAINING

fitur_terpilih = ['B1', 'B2', 'B3', 'B4', 'B5', 'B12']

data = out_df
X = data[fitur_terpilih] 
y = data['jenis_lahan']        

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict land cover types for the test data
y_pred = rf_classifier.predict(X_test)

# Create a DataFrame to store predicted and actual labels
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

# Export the results DataFrame to an Excel file
results_df.to_excel(script_directory + '/prediction_result/presentasi2/prediction_presentasi.xlsx', index=False)

#%% EVALUASI

df = results_df

def evaluation_function(prediction_array, truth_array, bands):
    
    global f1_air_max, f1_bangunan_max, f1_area_hijau_max
    global band_air, band_bangunan, band_area_hijau
    # Compute confusion matrix
    cm = confusion_matrix(truth_array, prediction_array)
    classes = set(truth_array.unique())
    
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
    print()


prediction_array = df['Predicted']
true_array = df['Actual']

evaluation_function(prediction_array, true_array, [""])

#%% CLIP DATASETNYA
test_geojson=[{
    "type": "Polygon",
    "coordinates": [
          [
            [
              107.60178147990769,
              -6.905727488511772
            ],
            [
              107.56077233312823,
              -6.961069947029458
            ],
            [
              107.61000487795548,
              -6.997605717170558
            ],
            [
              107.63186672099931,
              -6.963129384252156
            ],
            [
              107.63200310067765,
              -6.9221012699455144
            ],
            [
              107.60178147990769,
              -6.905727488511772
            ]
          ]
        ]
    }]


# print(test_geojson[0]["coordinates"][0])

polygon = Polygon(test_geojson[0]["coordinates"][0])
polygon_gdf = gpd.GeoDataFrame(geometry=[polygon])
# polygon_gdf = gpd.read_file(geojson_path + file) 
polygon_gdf.crs = "EPSG:4326"  # Assuming WGS84
# polygon_gdf = gpd.GeoDataFrame(geometry=[Polygon(my_geojson[0]["coordinates"])])
polygon_gdf_reprojected = polygon_gdf.to_crs(b2_src_20.crs)


clipped_b1, transform_b1 = mask(b1_src_20, polygon_gdf_reprojected.geometry, crop=True)
clipped_b2, transform_b2 = mask(b2_src_20, polygon_gdf_reprojected.geometry, crop=True)
clipped_b3, transform_b3 = mask(b3_src_20, polygon_gdf_reprojected.geometry, crop=True)
clipped_b4, transform_b4 = mask(b4_src_20, polygon_gdf_reprojected.geometry, crop=True)
clipped_b5, transform_b5 = mask(b5_src_20, polygon_gdf_reprojected.geometry, crop=True)
clipped_b6, transform_b6 = mask(b6_src_20, polygon_gdf_reprojected.geometry, crop=True)
clipped_b7, transform_b7 = mask(b7_src_20, polygon_gdf_reprojected.geometry, crop=True)
clipped_b8, transform_b8 = mask(b8A_src_20, polygon_gdf_reprojected.geometry, crop=True)
clipped_b11, transform_b11 = mask(b11_src_20, polygon_gdf_reprojected.geometry, crop=True)
clipped_b12, transform_b12 = mask(b12_src_20, polygon_gdf_reprojected.geometry, crop=True)

arr_of_clipped = [clipped_b1, clipped_b2, clipped_b3, clipped_b4, clipped_b5, clipped_b6, clipped_b7, clipped_b8, clipped_b11, clipped_b12]
print(clipped_b1)


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#%% PEMBUATAN DATASET TIDAK BERLABEL DARI DATA SATELIT

output_arr = [[],[],[],[],[],[],[],[],[],[],[], []]
done_xy = False

x_size = len(clipped_b1[0])
y_size = len(clipped_b1[0][0])

print(x_size, y_size)

for row_idx in range(0, x_size - 3, 3):
    for col_idx in range(0, y_size - 3, 3):
        band_ctr = 0
        
        x_start, y_start = row_idx, col_idx
        cluster_input = []
        for i in range (x_start, x_start + 3):
            for j in range(y_start, y_start + 3):
                temp = []
                for k in range(10):
                    temp.append(arr_of_clipped[k][0][i][j])
                cluster_input.append(temp)

        silhouette_scores = []
        max_k = 5  
    
        for k in range(2, max_k + 1):   
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(cluster_input)
            labels = kmeans.labels_
            if statistics.mode(labels) == 0:
                continue
            score = silhouette_score(cluster_input, labels)
            silhouette_scores.append(score)

        
        
        optimal_k = 1
        if(len(silhouette_scores) != 0):
            # print(np.max(silhouette_scores))
            optimal_k = np.argmax(silhouette_scores) + 2
        
        band_sum = [0,0,0,0,0,0,0,0,0,0]
        
        
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        labels = kmeans.fit_predict(cluster_input)
        majority = statistics.mode(labels)

        cnt = 0
                
        contain_zero = False
        for i in range(0, len(labels)):
            if labels[i] == majority and not contain_zero:
                curr_data = cluster_input[i]
                for j in range(0, 10):
                    if(curr_data[j] == 0):
                        contain_zero = True
                        continue
                    band_sum[j] += curr_data[j]
                cnt+=1
        
        if(statistics.mode(band_sum) == 0):
            continue
        for i in range(10):
            output_arr[i].append(band_sum[i]/cnt)
        
        output_arr[10].append(row_idx)
        output_arr[11].append(col_idx)
    
        print("done untuk grid ", row_idx, col_idx)
        

print("sudah selesai..")
#%% output ke excel

output_filename = 'dataset_prediksi_presentasi_3'
out_df = pd.DataFrame({'B1': output_arr[0], 'B2': output_arr[1], 'B3': output_arr[2], 'B4': output_arr[3], 'B5': output_arr[4], 'B6': output_arr[5], 'B7': output_arr[6], 'B8': output_arr[7], 'B11': output_arr[8], 'B12': output_arr[9], 'x': output_arr[10], 'y': output_arr[11]})

out_df.to_excel(script_directory + '/coba_remapping/' + output_filename + ".xlsx", index=False)

#%% KODE TRAINING UNTUK PREDIKSI(VICO VERSION)

def train_only(x_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    return rf_classifier

def predict_only(rfc, data, x_coor, y_coor):
    y_pred = rfc.predict(data)
    results_df = pd.DataFrame({'x': x_coor, 'y': y_coor, 'jenis_lahan': y_pred})
    nama_file = "hasil_prediksi_presentasi"
    results_df.to_excel(script_directory + '/prediction_result/real_predict/' + nama_file + '.xlsx', index=False)

    return y_pred

#%% KODE TRAINING COBA 20m

training_data = pd.read_excel(script_directory + '/output_labelling/presentasi' + '/dataset_presentasi_2.xlsx')

x_train = training_data[fitur_terpilih]
y_train = training_data['jenis_lahan']
    
rfc = train_only(x_train, y_train)

predict_data = pd.read_excel(script_directory + '/coba_remapping' + '/dataset_prediksi_presentasi_2.xlsx')

hasil = predict_only(rfc, predict_data[fitur_terpilih], predict_data['x'], predict_data['y'])

print(hasil)

#%% PETAKAN KEMBALI KE 2d

hasil_prediksi = pd.read_excel(script_directory +"/prediction_result/real_predict/hasil_prediksi_presentasi.xlsx")


normalized_b2 = clipped_b2[0] / clipped_b2[0].max() * 490
normalized_b3 = clipped_b3[0] / clipped_b3[0].max() * 460
normalized_b4 = clipped_b4[0] / clipped_b4[0].max() * 510


rgb_image = np.dstack((normalized_b4, normalized_b3, normalized_b2)).astype(np.uint8)
rgb_raw = np.dstack((clipped_b2[0], clipped_b3[0], clipped_b4[0]))

plt.figure(figsize=(20, 12))  # Set width to 10 inches, height to 6 inches
plt.imshow(rgb_image)
#%% print peta

lahan = hasil_prediksi['jenis_lahan']
x_all = hasil_prediksi['x']
y_all = hasil_prediksi['y']

hasil_b2 = clipped_b2[0].copy()
hasil_b3 = clipped_b3[0].copy()
hasil_b4 = clipped_b4[0].copy()


pixel_count = hasil_prediksi.shape[0]
for i in range(0, pixel_count):
    x = x_all[i]
    y = y_all[i]
    
    for j in range(x, x + 3):
        for k in range(y, y + 3):
            if lahan[i] == 'bangunan':
                hasil_b2[j][k] = 200
                hasil_b3[j][k] = 100
                hasil_b4[j][k] = 0
            elif lahan[i] == 'area_hijau':
                hasil_b2[j][k] = 0
                hasil_b3[j][k] = 200
                hasil_b4[j][k] = 0
            else:
                hasil_b2[j][k] = 0
                hasil_b3[j][k] = 0
                hasil_b4[j][k] = 200
        

rgb_image = np.dstack((hasil_b2, hasil_b3, hasil_b4)).astype(np.uint8)
plt.figure(figsize=(12, 12))  # Set width to 10 inches, height to 6 inches
plt.imshow(rgb_image)
    