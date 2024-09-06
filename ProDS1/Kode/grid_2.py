#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 14:35:35 2024

@author: MSI
"""

#%% GRID MAIN

import rasterio
import statistics

import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
import pandas as pd
from shapely import Polygon
import geopandas as gpd
import rasterio.crs as CRS
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from itertools import combinations
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns

script_directory = os.path.dirname(os.path.abspath(__file__))

#%% dataset load resolusi 20m
b1_path_20 = script_directory + '/jp2/20m/T48MYT_20231220T030131_B01_20m.jp2'
b2_path_20 = script_directory + '/jp2/20m/T48MYT_20231220T030131_B02_20m.jp2'
b3_path_20 = script_directory + '/jp2/20m/T48MYT_20231220T030131_B03_20m.jp2'
b4_path_20 = script_directory + '/jp2/20m/T48MYT_20231220T030131_B04_20m.jp2'
b5_path_20 = script_directory + '/jp2/20m/T48MYT_20231220T030131_B05_20m.jp2'
b6_path_20 = script_directory + '/jp2/20m/T48MYT_20231220T030131_B06_20m.jp2'
b7_path_20 = script_directory + '/jp2/20m/T48MYT_20231220T030131_B07_20m.jp2'
b8A_path_20 = script_directory + '/jp2/20m/T48MYT_20231220T030131_B8A_20m.jp2'
b11_path_20 = script_directory + '/jp2/20m/T48MYT_20231220T030131_B11_20m.jp2'
b12_path_20 = script_directory + '/jp2/20m/T48MYT_20231220T030131_B12_20m.jp2'

b1_src_20 = rasterio.open(b1_path_20)
b2_src_20 = rasterio.open(b2_path_20)
b3_src_20 = rasterio.open(b3_path_20)
b4_src_20 = rasterio.open(b4_path_20)
b5_src_20 = rasterio.open(b5_path_20)
b6_src_20 = rasterio.open(b6_path_20)
b7_src_20 = rasterio.open(b7_path_20)
b8A_src_20 = rasterio.open(b8A_path_20)
b11_src_20 = rasterio.open(b11_path_20)
b12_src_20 = rasterio.open(b12_path_20)

B1_20 = b1_src_20.read(1)
B2_20 = b2_src_20.read(1)
B3_20 = b3_src_20.read(1)
B4_20 = b4_src_20.read(1)
B5_20 = b5_src_20.read(1)
B6_20 = b6_src_20.read(1)
B7_20 = b7_src_20.read(1)
B8A_20 = b8A_src_20.read(1)
B11_20 = b11_src_20.read(1)
B12_20 = b12_src_20.read(1)

#%%
grid_dir = script_directory + "/geojson/label_by_grid_"

jumlah_labeled_file = 2

B1 = []
B2 = []
B3 = []
B4 = []
B5 = []
B6 = []
B7 = []
B8 = []
B11 = []
B12 = []
label_output = []


for i in range(1, jumlah_labeled_file+1):
    grid_gdf = gpd.read_file(grid_dir+str(i)+".geojson") 
    grid_gdf.crs = "EPSG:4326"  
    
    grid_gdf = grid_gdf.to_crs(b1_src_20.crs)
    band_list = [B1_20, B2_20, B3_20, B4_20, B5_20, B6_20, B7_20, B8A_20, B11_20, B12_20]
    
    
    label = ""
    max_num = 9999999999
    
    # print(band_list[0][1])
    
    for index, row in grid_gdf.iterrows():
        multipoly = row['geometry']
        
        if index == 0:
            label = "bangunan"
        elif index == 1:
            label = "area_hijau"
        else:
            label = "air"
            
        for poly in multipoly.geoms:
            
            cluster_input = []
            # print(poly)
            coords = poly.exterior.coords
            
        
            xmin, ymin = max_num, max_num
            xmax, ymax = 0, 0
            
            for point in coords:
                # print(point)
                x_raster, y_raster = b1_src_20.index(point[0], point[1])
                if(x_raster < xmin):
                    xmin = x_raster
                if(x_raster > xmax):
                    xmax = x_raster
                if(y_raster < ymin):
                    ymin = y_raster
                if(y_raster > ymax):
                    ymax = y_raster
    
            selisihX = xmax-xmin
            selisihY = ymax-ymin
            
            if (selisihX <= 5 and selisihX >= 3 and selisihY <= 5 and selisihY >= 3):
                for i in range(xmin, xmax+1):
                    for j in range(ymin, ymax+1):
                        temp = []
                        for k in range(0, 10):
                            try:
                                temp.append(band_list[k][i][j])
                            except:
                                print("index out of bound")
                        
                        cluster_input.append(temp)
                        # print(selisihX, selisihY)
                # print(cluster_input)
                # print("process cluster")
                
                #coba clustering
                #tentukan k dulu menggunakan silhouette score
                silhouette_scores = []
                max_k = 5
                
                for k in range(2, max_k + 1):   
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(cluster_input)
                    labels = kmeans.labels_
                    score = silhouette_score(cluster_input, labels)
                    silhouette_scores.append(score)
                
                optimal_k = np.argmax(silhouette_scores) + 2
                
                band_sum = [0,0,0,0,0,0,0,0,0,0]
                
                #setelah dpt k_optimal, pakai untuk clustering
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                labels = kmeans.fit_predict(cluster_input)
                # print(len(labels))
                # print(len(cluster_input))
                majority = statistics.mode(labels)
                cnt = 0
                
                # print(len(labels))
                
                for i in range(0, len(labels)):
                    if labels[i] == majority:
                        curr_data = cluster_input[i]
                        for j in range(0, 10):
                            # print(curr_data)
                            band_sum[j] += curr_data[j]
                        cnt+=1
                
                B1.append(band_sum[0]/cnt)
                B2.append(band_sum[1]/cnt)
                B3.append(band_sum[2]/cnt)
                B4.append(band_sum[3]/cnt)
                B5.append(band_sum[4]/cnt)
                B6.append(band_sum[5]/cnt)
                B7.append(band_sum[6]/cnt)
                B8.append(band_sum[7]/cnt)
                B11.append(band_sum[8]/cnt)
                B12.append(band_sum[9]/cnt)
                label_output.append(label)
        
        
#%% output ke excel

output_filename = 'dataset_grid_metode_2'
out_df = pd.DataFrame({
    'B1': B1,
    'B2': B2,
    'B3': B3,
    'B4': B4,
    'B5': B5,
    'B6': B6,
    'B7': B7,
    'B8': B8,
    'B11': B11,
    'B12': B12,
    'jenis_lahan': label_output
    })

out_df = out_df.drop_duplicates()

print(out_df.shape)

out_df.to_excel(script_directory + '/output_labelling/' + output_filename + ".xlsx", index=False)
        
#%% KODE TRAINING (MARK VERSION)

# Load the data from the Excel file

data = pd.read_excel(script_directory + '/output_labelling' + '/dataset_grid_metode_2.xlsx')

# Split the data into features (spectral bands) and target label


# X = data[['B1', 'B2', 'B4', 'B5', 'B11', 'B12']]  # Features: Spectral bands B2, B3, B4
# X = data[['B4', 'B6', 'B7', 'B8', 'B11', 'B12']]
X = data[['B1', 'B2', 'B3', 'B4', 'B5', 'B12']] # metode 2
# X = data[['B1', 'B2', 'B3', 'B4', 'B6', 'B12']] #metode 2x2 cluster

# X = data[['B1_min', 'B1_mean', 'B1_max', 'B2_mean', 'B4_min', 'B4_mean', 'B4_max',
#        'B12_min', 'B12_mean', 'B12_max']]

y = data['jenis_lahan']        # Target label: Land cover types

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict land cover types for the test data
y_pred = rf_classifier.predict(X_test)

# Create a DataFrame to store predicted and actual labels
results_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})

# Export the results DataFrame to an Excel file
results_df.to_excel(script_directory + '/prediction_results.xlsx', index=False)

#%% read excel dan masukan ke variabel (CONTOH)

df = pd.read_excel(script_directory + "/prediction_results.xlsx")

prediction_array = df['Predicted']
true_array = df['Actual']


#%% function buat matrix (andrea version)

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

evaluation_function(prediction_array, true_array, [""])
#%% coba di data asli

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


print(test_geojson[0]["coordinates"][0])

polygon = Polygon(test_geojson[0]["coordinates"][0])
polygon_gdf = gpd.GeoDataFrame(geometry=[polygon])
# polygon_gdf = gpd.read_file(geojson_path + file) 
polygon_gdf.crs = "EPSG:4326"  # Assuming WGS84
# polygon_gdf = gpd.GeoDataFrame(geometry=[Polygon(my_geojson[0]["coordinates"])])
polygon_gdf_reprojected = polygon_gdf.to_crs(b2_src_20.crs)

# %% clipping all bands

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

#%%

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

#%% pembuatan dataset

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

output_filename = 'dataset_prediksi_grid_3x3'
out_df = pd.DataFrame({'B1': output_arr[0], 'B2': output_arr[1], 'B3': output_arr[2], 'B4': output_arr[3], 'B5': output_arr[4], 'B6': output_arr[5], 'B7': output_arr[6], 'B8': output_arr[7], 'B11': output_arr[8], 'B12': output_arr[9], 'x': output_arr[10], 'y': output_arr[11]})

out_df.to_excel(script_directory + '/coba_remapping/' + output_filename + ".xlsx", index=False)

#%% KODE TRAINING UNTUK PREDIKSI(VICO VERSION)

def train_only(x_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    return rf_classifier

#%% PREDIKSI ONLY

def predict_only(rfc, data, x_coor, y_coor):
    y_pred = rfc.predict(data)
    results_df = pd.DataFrame({'x': x_coor, 'y': y_coor, 'jenis_lahan': y_pred})
    nama_file = "hasil_prediksi_grid_3x3"
    results_df.to_excel(script_directory + '/prediction_result/real_predict/' + nama_file + '.xlsx', index=False)

    return y_pred

#%% KODE TRAINING COBA 20m

training_data = pd.read_excel(script_directory + '/output_labelling' + '/dataset_grid_metode_2.xlsx')

#data kalo baseline yg ini
# x = ['B1', 'B2', 'B6', 'B8', 'B12']
# x = ['B1', 'B2', 'B3', 'B4', 'B6', 'B12'] #metode 2x2 cluster
# x = ['B1_min', 'B1_mean', 'B2_mean', 'B3_mean', 'B4_min', 'B4_mean',
#         'B4_max', 'B12_min', 'B12_mean', 'B12_max']

x = ['B1', 'B2', 'B3', 'B4', 'B5', 'B12'] # metode 2
# x = ['B2_mean', 'B4_mean', 'B12_min', 'B12_mean', 'B12_max']
x_train = training_data[x]
y_train = training_data['jenis_lahan']
    
rfc = train_only(x_train, y_train)

#%% prediksi sesungguhnya

predict_data = pd.read_excel(script_directory + '/coba_remapping' + '/dataset_prediksi_grid_3x3.xlsx')
print(predict_data)


#%% prediksi

hasil = predict_only(rfc, predict_data[x], predict_data['x'], predict_data['y'])

print(hasil)

#%% PETAKAN KEMBALI KE 2d

hasil_prediksi = pd.read_excel(script_directory +"/prediction_result/real_predict/hasil_prediksi_grid_3x3.xlsx")

# %% show rgb

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
    