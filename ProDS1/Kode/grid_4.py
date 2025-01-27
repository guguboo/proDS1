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
import addFeature as af


script_directory = os.path.dirname(os.path.abspath(__file__))
script_directory =  os.path.dirname(script_directory)

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
grid_dir = script_directory + "/geojson/label_2x2_"

labelled = 2
    
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
NDVI_min = []
NDVI_max = []
NDVI_mean = []
EVI_min = []
EVI_max = []
EVI_mean = []
labels_output = []

for i in range(1,labelled+1):
    
    grid_gdf = gpd.read_file(grid_dir + str(i) + ".geojson") 
    grid_gdf.crs = "EPSG:4326"  
    
    grid_gdf = grid_gdf.to_crs(b1_src_20.crs)
    
    # print(grid_gdf.shape)

    
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
            NDVI_min.append(af.addNDVI(b4_min, b8_min))
            NDVI_mean.append(af.addNDVI(b4_sum/cnt, b8_sum/cnt))
            NDVI_max.append(af.addNDVI(b4_max, b8_max))
            EVI_min.append(af.addEVI(b2_min,b4_min, b8_min))
            EVI_mean.append(af.addEVI(b2_sum/cnt,b4_sum/cnt, b8_sum/cnt))
            EVI_max.append(af.addEVI(b2_max,b4_max, b8_max))
            labels_output.append(label)
                
        #%% output ke excel

        output_filename = 'dataset_grid_statistik_2x2'
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
            'NDVI_min': NDVI_min, 'NDVI_mean': NDVI_mean, 'NDVI_max': NDVI_max,       
            'EVI_min': EVI_min, 'EVI_mean': EVI_mean, 'EVI_max': EVI_max, 
            'jenis_lahan': labels_output})

        out_df = out_df.drop_duplicates()


        out_df.to_excel(script_directory + '/output_labelling/' + output_filename + ".xlsx", index=False)
        
        
        
# polygon_gdf = gpd.GeoDataFrame(geometry=[Polygon(my_geojson[0]["coordinates"])])
# polygon_gdf_reprojected = polygon_gdf.to_crs(b2_src_20.crs)

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
NDVI_min = []
NDVI_max = []
NDVI_mean = []
EVI_min = []
EVI_max = []
EVI_mean = []
x = []
y = []

x_size = len(clipped_b1[0])
y_size = len(clipped_b1[0][0])

cnt = 9

for row_idx in range(0, x_size - 2, 2):
    for col_idx in range(0, y_size - 2, 2):
        b1_sum, b2_sum, b3_sum, b4_sum, b5_sum, b6_sum, b7_sum, b8_sum, b11_sum, b12_sum = 0,0,0,0,0,0,0,0,0,0
        b1_min, b2_min, b3_min, b4_min, b5_min, b6_min, b7_min, b8_min, b11_min, b12_min = max_num, max_num, max_num, max_num, max_num, max_num, max_num, max_num, max_num, max_num
        b1_max, b2_max, b3_max, b4_max, b5_max, b6_max, b7_max, b8_max, b11_max, b12_max = 0,0,0,0,0,0,0,0,0,0
        
        for i in range(row_idx, row_idx+2):
            for j in range(col_idx, col_idx+2):
                try:
                    b1_sum += clipped_b1[0][i][j]
                    b2_sum += clipped_b2[0][i][j]
                    b3_sum += clipped_b3[0][i][j]
                    b4_sum += clipped_b4[0][i][j]
                    b5_sum += clipped_b5[0][i][j]
                    b6_sum += clipped_b6[0][i][j]
                    b7_sum += clipped_b7[0][i][j]
                    b8_sum += clipped_b8[0][i][j]
                    b11_sum += clipped_b11[0][i][j]
                    b12_sum += clipped_b12[0][i][j]        
                    
                    if B1_20[i][j] > b1_max:
                        b1_max = clipped_b1[0][i][j]
                    if B2_20[i][j] > b2_max:
                        b2_max = clipped_b2[0][i][j]  
                    if B3_20[i][j] > b3_max:
                        b3_max = clipped_b3[0][i][j]
                    if B4_20[i][j] > b4_max:
                        b4_max = clipped_b4[0][i][j]    
                    if B5_20[i][j] > b5_max:
                        b5_max = clipped_b5[0][i][j]
                    if B6_20[i][j] > b6_max:
                        b6_max = clipped_b6[0][i][j]
                    if B7_20[i][j] > b7_max:
                        b7_max = clipped_b7[0][i][j]
                    if B8A_20[i][j] > b8_max:
                        b8_max = clipped_b8[0][i][j]
                    if B11_20[i][j] > b11_max:
                        b11_max = clipped_b11[0][i][j]
                    if B12_20[i][j] > b12_max:
                        b12_max = clipped_b12[0][i][j]
                
                    if B1_20[i][j] < b1_min:
                        b1_min = clipped_b1[0][i][j]
                    if B2_20[i][j] < b2_min:
                        b2_min = clipped_b2[0][i][j]  
                    if B3_20[i][j] < b3_min:
                        b3_min = clipped_b3[0][i][j]
                    if B4_20[i][j] < b4_min:
                        b4_min = clipped_b4[0][i][j]    
                    if B5_20[i][j] < b5_min:
                        b5_min = clipped_b5[0][i][j]
                    if B6_20[i][j] < b6_min:
                        b6_min = clipped_b6[0][i][j]
                    if B7_20[i][j] < b7_min:
                        b7_min = clipped_b7[0][i][j]
                    if B8A_20[i][j] < b8_min:
                        b8_min = clipped_b8[0][i][j]
                    if B11_20[i][j] < b11_min:
                        b11_min = clipped_b11[0][i][j]
                    if B12_20[i][j] < b12_min:
                        b12_min = clipped_b12[0][i][j]
                except:
                    ""
        if(b1_min != 0):
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
            NDVI_min.append(af.addNDVI(b4_min, b8_min))
            NDVI_mean.append(af.addNDVI(b4_sum/cnt, b8_sum/cnt))
            NDVI_max.append(af.addNDVI(b4_max, b8_max))
            EVI_min.append(af.addEVI(b2_min,b4_min, b8_min))
            EVI_mean.append(af.addEVI(b2_sum/cnt,b4_sum/cnt, b8_sum/cnt))
            EVI_max.append(af.addEVI(b2_max,b4_max, b8_max))
            x.append(row_idx)
            y.append(col_idx)
            print(f"row ke-{i} col ke-{j}")
                
output_filename = 'dataset_prediksi_statistik_2x2'
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
    'NDVI_min': NDVI_min, 'NDVI_mean': NDVI_mean, 'NDVI_max': NDVI_max,  
    'EVI_min': EVI_min, 'EVI_mean': EVI_mean, 'EVI_max': EVI_max, 
    'x': x, 'y': y
    })

print("dropping dupes")
out_df = out_df.drop_duplicates()


print("creating dataset")
out_df.to_excel(script_directory + '/coba_remapping/' + output_filename + ".xlsx", index=False)

print("dataset done")

#%%

hasil_prediksi = pd.read_excel(script_directory +"/prediction_result/real_predict/hasil_prediksi_statistik_2x2.xlsx")

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
    
    for j in range(x, x + 2):
        for k in range(y, y + 2):
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
    
