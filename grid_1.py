#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 07:16:27 2024

@author: MSI
"""

#%% GRID MAIN

import rasterio
import os
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

print(grid_gdf.shape)

B1_min = []
B1_mean = []
B1_max = []
B2_min = []
B2_mean = []
B2_max = []
B3_min = []
B3_mean = []
B3_max = []
B4_min = []
B4_mean = []
B4_max = []
B5_min = []
B5_mean = []
B5_max = []
B6_min = []
B6_mean = []
B6_max = []
B7_min = []
B7_mean = []
B7_max = []
B8_min = []
B8_mean = []
B8_max = []
B11_min = []
B11_mean = []
B11_max = []
B12_min = []
B12_mean = []
B12_max = []
labels_output = []


label = ""
max_num = 9999999999

for index, row in grid_gdf.iterrows():
    multipoly = row['geometry']
    
    if index == 0:
        label = "bangunan"
    elif index == 1:
        label = "area_hijau"
    else:
        label = "air"
        
    for poly in multipoly.geoms:
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
        
        b1_sum, b2_sum, b3_sum, b4_sum, b5_sum, b6_sum, b7_sum, b8_sum, b11_sum, b12_sum = 0,0,0,0,0,0,0,0,0,0
        b1_min, b2_min, b3_min, b4_min, b5_min, b6_min, b7_min, b8_min, b11_min, b12_min = max_num, max_num, max_num, max_num, max_num, max_num, max_num, max_num, max_num, max_num
        b1_max, b2_max, b3_max, b4_max, b5_max, b6_max, b7_max, b8_max, b11_max, b12_max = 0,0,0,0,0,0,0,0,0,0
        cnt = 0
        
        for i in range(xmin, xmax+1):
            for j in range(ymin, ymax+1):
                try:
                    b1_sum += B1_20[i][j]
                    b2_sum += B2_20[i][j]
                    b3_sum += B3_20[i][j]
                    b4_sum += B4_20[i][j]
                    b5_sum += B5_20[i][j]
                    b6_sum += B6_20[i][j]
                    b7_sum += B7_20[i][j]
                    b8_sum += B8A_20[i][j]
                    b11_sum += B11_20[i][j]
                    b12_sum += B12_20[i][j]
                    
                    cnt += 1
                    
                    if B1_20[i][j] > b1_max:
                        b1_max = B1_20[i][j]
                    if B2_20[i][j] > b2_max:
                        b2_max = B2_20[i][j]  
                    if B3_20[i][j] > b3_max:
                        b3_max = B3_20[i][j]
                    if B4_20[i][j] > b4_max:
                        b4_max = B4_20[i][j]    
                    if B5_20[i][j] > b5_max:
                        b5_max = B5_20[i][j]
                    if B6_20[i][j] > b6_max:
                        b6_max = B6_20[i][j]
                    if B7_20[i][j] > b7_max:
                        b7_max = B7_20[i][j]
                    if B8A_20[i][j] > b8_max:
                        b8_max = B8A_20[i][j]
                    if B11_20[i][j] > b11_max:
                        b11_max = B11_20[i][j]
                    if B12_20[i][j] > b12_max:
                        b12_max = B12_20[i][j]
                
                    if B1_20[i][j] < b1_min:
                        b1_min = B1_20[i][j]
                    if B2_20[i][j] < b2_min:
                        b2_min = B2_20[i][j]  
                    if B3_20[i][j] < b3_min:
                        b3_min = B3_20[i][j]
                    if B4_20[i][j] < b4_min:
                        b4_min = B4_20[i][j]    
                    if B5_20[i][j] < b5_min:
                        b5_min = B5_20[i][j]
                    if B6_20[i][j] < b6_min:
                        b6_min = B6_20[i][j]
                    if B7_20[i][j] < b7_min:
                        b7_min = B7_20[i][j]
                    if B8A_20[i][j] < b8_min:
                        b8_min = B8A_20[i][j]
                    if B11_20[i][j] < b11_min:
                        b11_min = B11_20[i][j]
                    if B12_20[i][j] < b12_min:
                        b12_min = B12_20[i][j]
                except:
                    ""
                    
        B1_min.append(b1_min)
        B1_mean.append(b1_sum / cnt)
        B1_max.append(b1_max)  
        B2_min.append(b2_min)
        B2_mean.append(b2_sum / cnt)
        B2_max.append(b2_max)  
        B3_min.append(b3_min)
        B3_mean.append(b3_sum / cnt)
        B3_max.append(b3_max)  
        B4_min.append(b4_min)
        B4_mean.append(b4_sum / cnt)
        B4_max.append(b4_max)  
        B5_min.append(b5_min)
        B5_mean.append(b5_sum / cnt)
        B5_max.append(b5_max)  
        B6_min.append(b6_min)
        B6_mean.append(b6_sum / cnt)
        B6_max.append(b6_max)  
        B7_min.append(b7_min)
        B7_mean.append(b7_sum / cnt)
        B7_max.append(b7_max)  
        B8_min.append(b8_min)
        B8_mean.append(b8_sum / cnt)
        B8_max.append(b8_max)  
        B11_min.append(b11_min)
        B11_mean.append(b11_sum / cnt)
        B11_max.append(b11_max)  
        B12_min.append(b12_min)
        B12_mean.append(b12_sum / cnt)
        B12_max.append(b12_max)  
        labels_output.append(label)
                
        #%% output ke excel


        output_counter = 1
        done_output = False
        output_filename = 'coba_coba_20m'
        out_df = pd.DataFrame({
            'B1_min': B1_min, 'B1_mean': B1_mean, 'B1_max': B1_max, 
            'B2_min': B2_min, 'B2_mean': B2_mean, 'B2_max': B2_max, 
            'B3_min': B3_min, 'B3_mean': B3_mean, 'B3_max': B3_max, 
            'B4_min': B4_min, 'B4_mean': B4_mean, 'B4_max': B4_max, 
            'B5_min': B5_min, 'B5_mean': B5_mean, 'B5_max': B5_max, 
            'B6_min': B6_min, 'B6_mean': B6_mean, 'B6_max': B6_max, 
            'B7_min': B7_min, 'B7_mean': B7_mean, 'B7_max': B7_max, 
            'B8_min': B8_min, 'B8_mean': B8_mean, 'B8_max': B8_max, 
            'B11_min': B11_min, 'B11_mean': B11_mean, 'B11_max': B11_max, 
            'B12_min': B12_min, 'B12_mean': B12_mean, 'B12_max': B12_max,             
            'jenis_lahan': labels_output})

        out_df = out_df.drop_duplicates()


        out_df.to_excel(script_directory + '/output_labelling/' + output_filename + "_" + str(output_counter) + ".xlsx", index=False)
        done_output = True

        
        
# polygon_gdf = gpd.GeoDataFrame(geometry=[Polygon(my_geojson[0]["coordinates"])])
# polygon_gdf_reprojected = polygon_gdf.to_crs(b2_src_20.crs)
