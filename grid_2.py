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
grid_dir = script_directory + "/geojson/label_by_grid_1.geojson"

grid_gdf = gpd.read_file(grid_dir) 
grid_gdf.crs = "EPSG:4326"  

grid_gdf = grid_gdf.to_crs(b1_src_20.crs)
band_list = [B1_20, B2_20, B3_20, B4_20, B5_20, B6_20, B7_20, B8A_20, B11_20, B12_20]

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

        if (xmax - xmin <= 5 and ymax - ymin <= 5):
            # print(xmax-xmin)
            # print(ymax-ymin)
            for i in range(xmin, xmax+1):
                for j in range(ymin, ymax+1):
                    temp = []
                    for k in range(0, 10):
                        try:
                            temp.append(band_list[k][i][j])
                        except:
                            print("index out of bound")
                    
                    cluster_input.append(temp)
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
        kmeans = KMeans(n_clusters=k, random_state=42)
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


out_df.to_excel(script_directory + '/output_labelling/' + output_filename + ".xlsx", index=False)
        