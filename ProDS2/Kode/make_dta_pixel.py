#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 07:55:43 2024

@author: MSI
"""


import rasterio
import os
import sys
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)
import addFeature as af
import time

def make_all_dta():
    start_time = time.time()
    bands_src = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    with open(script_dir + '/last_fetched.txt', 'r') as file:
        date_filename = file.read().strip()

    band_path = parent_dir + f'/Data/satelit/{date_filename}/'

    for i in range(1, 13):
        if i != 9 and i != 10:
            curr_band = rasterio.open(band_path + "B" + str(i) + ".jp2")
            bands_src[i] = curr_band

    bands_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for i in range(1, 13):
        if i != 9 and i != 10:
            bands_list[i] = bands_src[i].read(1)

    print("Successfully Read Bands Raster")

    
    # print(test_geojson[0]["coordinates"][0])
    # polygon_gdf = gpd.read_file(geojson_path + file) 
    # polygon_gdf = gpd.GeoDataFrame(geometry=[Polygon(my_geojson[0]["coordinates"])])


    DTA = gpd.read_file(script_dir + "/mygeodata.zip")
    real_names = []
    for idx, dta in DTA.iterrows():
        dupe_idx = 2
        if dta['name'] in real_names:    
            real_names.append(dta['name'] + str(dupe_idx))
            name = dta['name'] + "_" + str(dupe_idx) + "_pixel.xlsx"
            name = name.replace("/", "_")
            name = name.replace(" ", "")
            dupe_idx += 1
        else:
            real_names.append(dta['name'])
            name = dta['name'] + "_pixel.xlsx"
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
    
            # print(clipped_bands)            
            clip = clipped_bands[1]
            for row_idx in range(0,len(clip)):
                for col_idx in range(0, len(clip[row_idx])):
                    curr_sum = 0
                    for i in range(1, 13):
                        if i != 9 and i != 10:
                            curr_sum += clipped_bands[i][row_idx][col_idx]
                    
                    if curr_sum != 0:
                        for i in range(1, 13):
                            if i != 9 and i != 10:
                                output_arr[i].append(clipped_bands[i][row_idx][col_idx])
                                
                        output_arr[13].append(row_idx)
                        output_arr[14].append(col_idx)
                            

            for arr in output_arr:
                print(len(arr))
            
            
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
        end_time = time.time()
        elapsed_time = end_time - start_time
    return f"DTA files created in {elapsed_time:.2f} seconds."

make_all_dta()