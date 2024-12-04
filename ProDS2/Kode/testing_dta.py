#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:51:39 2024

@author: MSI
"""


import rasterio
import os
import sys
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from rasterio.mask import mask

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)
import addFeature as af
import time


bands_src = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

band_path = parent_dir + '/Data/satelit/21082024/'

for i in range(1, 13):
    if i != 9 and i != 10:
        curr_band = rasterio.open(band_path + "B" + str(i) + ".jp2")
        bands_src[i] = curr_band

bands_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(1, 13):
    if i != 9 and i != 10:
        bands_list[i] = bands_src[i].read(1)

print("Successfully Read Bands Raster")

#%%
# print(test_geojson[0]["coordinates"][0])
# polygon_gdf = gpd.read_file(geojson_path + file) 
# polygon_gdf = gpd.GeoDataFrame(geometry=[Polygon(my_geojson[0]["coordinates"])])


DTA = gpd.read_file(script_dir + "/mygeodata.zip")

src = bands_src[1]
for idx, dta in DTA.iterrows():
    name = dta['name'] + ".xlsx"
    name = name.replace("/", "_")
    name = name.replace(" ", "")
    print(f"Processing {dta['name']}")
    file_path = os.path.join(parent_dir + '/Data/(to predict)/', name)
    if os.path.isfile(file_path):
        print("File Existed.")
    else:
          
        polygon_gdf = gpd.GeoDataFrame(geometry=[dta['geometry']])
        polygon_gdf.crs = "EPSG:4326"
    
        polygon_gdf_reprojected = polygon_gdf.to_crs(bands_src[1].crs)
        
        clipped_bands = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        
        for i in range(1, 13):
            if i != 9 and i != 10:
                clipped, transformed = mask(bands_src[i], polygon_gdf_reprojected.geometry, crop=True)
                clipped_bands[i] = clipped[0]
        
        
        output_arr = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        done_xy = False
        length_max = 10000000
        
        # print(clipped_bands)
        for i in range(1, 13):
            if i != 9 and i != 10:
                print("clipping band ke-" + str(i))
                clip = clipped_bands[i]
                for row_idx in range(0,len(clip)):
                    clip_row = clip[row_idx]
                    for col_idx in range(0, len(clip_row)):
                        item = clip_row[col_idx]
                        if item != 0:
                            pixel_area = 0
                            cnt = 0
                            for j in range(row_idx-1, row_idx+2):
                                for k in range(col_idx-1, col_idx+2):
                                    try:
                                        pixel_area += bands_list[i][j][k] 
                                        cnt += 1
                                    except:
                                        ""
                            if  len(output_arr[i]) < length_max:
                                output_arr[i].append(pixel_area/cnt)
                                if not done_xy:
                                    output_arr[13].append(row_idx)
                                    output_arr[14].append(col_idx)
                
                if not done_xy:
                    length_max = len(output_arr[13])
                done_xy = True
                
        
        print("Done clipping bands")

        out_df = pd.DataFrame({'B1': output_arr[1],
                               'B2': output_arr[2], 
                               'B3': output_arr[3], 
                               'B4': output_arr[4], 
                               'B5': output_arr[5], 
                               'B6': output_arr[6], 
                               'B7': output_arr[7], 
                               'B8': output_arr[8], 
                               'B11': output_arr[11],
                               'B12': output_arr[12],
                               'x': output_arr[13],
                               'y': output_arr[14]})
        
        out_df['NDVI'] = af.addNDVI(out_df['B4'], out_df['B8'])
        out_df['EVI'] = af.addEVI(out_df['B2'], out_df['B4'], out_df['B8'])
        out_df['NDWI'] = af.addNDWI(out_df['B3'], out_df['B8'])
        
        
        out_df.to_excel(parent_dir + '/Data/(to predict)/' + name, index=False)
        print(f"Successfully exported {name}")