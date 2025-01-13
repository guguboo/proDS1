#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:33:36 2024

@author: MSI
"""


import rasterio
import os
import sys
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from rasterio.transform import xy
from rasterio.io import MemoryFile
from pyproj import Transformer
from shapely.geometry import box, mapping

import numpy as np
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)

import time



def predict_all():
    for i in range(59):
        prediction(i)

def prediction(dta_idx):
    start_time = time.time()
    with open(script_dir + '/last_fetched.txt', 'r') as file:
        date_filename = file.read().strip()

    dta_df = pd.read_csv(script_dir + "/dta_filenames.csv")
    dta_filename = dta_df['dta_filenames'][dta_idx]
    
    labeling_filename = dta_filename.replace(".xlsx", "_pixel.xlsx")
        
    save_dir = os.path.join(parent_dir, 'Images')

    features_stage_1 = ["B1", "B5", "B11", "B12"]

    labeled = parent_dir + "/Labeled/Integration/"

    try:    
        df = pd.read_excel(labeled+date_filename+".xlsx")
        # print(df)
    except:
        raise Exception(f"Data training {date_filename} tidak ada")

    X = df.copy()
    y = df['land_cover']        


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



    rfc_1 = RandomForestClassifier(
        bootstrap=True,
        max_depth=20,
        max_features = 'sqrt',
        min_samples_leaf=1,
        min_samples_split=5,
        n_estimators=300,
        random_state=42  # To ensure reproducibility
    )

    rfc_1.fit(X[features_stage_1], grouped_land_cover)
    
    predict_data = pd.read_excel(parent_dir + '/Data/(to predict)/' + labeling_filename)
    
    hasil_1 = rfc_1.predict(predict_data[features_stage_1])

    # print(len(hasil_1)) 

    # print(sum(hasil_1=="group1"))
    df_g1 = predict_data.iloc[hasil_1=="group1"]
    df_g2 = predict_data.iloc[hasil_1=="group2"]
    df_g3 = predict_data.iloc[hasil_1=="group3"]
    
    g1_exist = True
    g2_exist = True
    g3_exist = True
    # print(df_g1['land_cover'].value_counts(),end="\n\n")
    # print(df_g2['land_cover'].value_counts(),end="\n\n")
    # print(df_g3['land_cover'].value_counts(),end="\n\n")

    end_time = time.time()


    # TAHAP 2 GROUP 1
    features_stage_2 = ["NDWI", 'NDVI', 'B3', 'B4', 'B5', 'B8', 'B11']
    X = df[df['land_cover'].isin(group_1)]
    y = X['land_cover']
    
    
    if(df_g1.shape[0] > 0):
        rfc_group1_1 = RandomForestClassifier(
            bootstrap=True,
            max_depth=20,
            max_features = 'sqrt',
            min_samples_leaf=1,
            min_samples_split=5,
            n_estimators=300,
            random_state=42  # To ensure reproducibility
        )
        
        rfc_group1_1.fit(X[features_stage_2], y)
    
        hasil_group1_1 = rfc_group1_1.predict(df_g1[features_stage_2])
        res_group_1 = df_g1.copy()
        res_group_1['prediction'] = hasil_group1_1
    else:
        res_group_1 = None
        g1_exist = False

    # TAHAP 2 GROUP 3

    features_stage_3 = ['B1', 'B2', 'B4', 'B5', 'B11', 'B12']
    X = df[df['land_cover'].isin(group_3)]
    y = X['land_cover']

    if(df_g2.shape[0] > 0):
        rfc_group3_1 = RandomForestClassifier(
            bootstrap=True,
            max_depth=20,
            max_features = 'sqrt',
            min_samples_leaf=1,
            min_samples_split=5,
            n_estimators=300,
            random_state=42  # To ensure reproducibility
        )
    
        rfc_group3_1.fit(X[features_stage_3], y)
    
        hasil_group3_1 = rfc_group3_1.predict(df_g3[features_stage_3])
        res_group_3 = df_g3.copy()
        res_group_3['prediction'] = hasil_group3_1
    else:
        res_group_3 = None
        g3_exist = False

        


    # TAHAP 2 GROUP 2
    
    features_stage_2 = ['B5', 'B7', 'B8', 'B12']
    X = df[df['land_cover'].isin(group_2)]
    y = X['land_cover']
    
    if(df_g3.shape[0] > 0):
        rfc_group2_1 = RandomForestClassifier(
            bootstrap=True,
            max_depth=20,
            max_features = 'sqrt',
            min_samples_leaf=1,
            min_samples_split=5,
            n_estimators=300,
            random_state=42  # To ensure reproducibility
        )
    
        rfc_group2_1.fit(X[features_stage_2], y)
    
        hasil_group2_1 = rfc_group2_1.predict(df_g2[features_stage_2])
        res_group_2 = df_g2.copy()
        res_group_2['prediction'] = hasil_group2_1
    else:
        res_group_2 = None
        g2_exist = False

    print(f"Time taken to predict: {end_time - start_time:.2f} seconds")
    
    # visualize
    result = [res for res, exist in zip([res_group_1, res_group_2, res_group_3], [g1_exist, g2_exist, g3_exist]) if exist]
    hasil = pd.concat(result, axis=0)
    hasil.head()

    lahan = hasil['prediction']
    x_all = hasil['x']
    y_all = hasil['y']


    bands_src = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    band_path = parent_dir + '/Data/satelit/21082024/'
    src_meta = 0
    src_transform = 0
    src_height = 0
    src_width = 0
    

    for i in range(1, 13):
        if i == 2 or i == 3 or i == 4:
            curr_band = rasterio.open(band_path + "B" + str(i) + ".jp2")
            src_meta = curr_band.meta
            src_transform = curr_band.transform
            src_height, src_width = curr_band.shape
            # print(src_meta)
            # print(src_transform)
            # print(curr_band.shape)
            bands_src[i] = curr_band
            print(curr_band)
        
            # break

    print("Successfully Read Bands Raster")
    
    matrix = np.zeros((src_height, src_width), dtype=src_meta["dtype"])
    new_meta = src_meta.copy()
    new_meta.update({
        "height": matrix.shape[0],
        "width": matrix.shape[1],
        "count": 1,
        "dtype": matrix.dtype,
        "transform": src_transform,  # Retain original georeferencing
    })
    
    for i in range(13, 17):
        memfile = MemoryFile()

        dataset = memfile.open(**new_meta)
        dataset.write(matrix, 1)
        bands_src[i] = dataset
        # print(dataset)
    
    
    DTA = gpd.read_file(script_dir + "/mygeodata.zip")
    
    src = bands_src[2]
    
    # print(src)
    real_names = []
    for idx, dta in DTA.iterrows():
        dupe_idx = 2
        
        if dta['name'] not in real_names:
            name = dta['name'] + "_pixel.xlsx"
            name = name.replace("/", "_")
            name = name.replace(" ", "")
            real_names.append(dta['name'])
        else:
            name = dta['name'] + "_" + str(dupe_idx) + "_pixel.xlsx"
            name = name.replace("/", "_")
            name = name.replace(" ", "")
            real_names.append(dta['name'])
            dupe_idx += 1
            
        if name.replace("_pixel.xlsx", ".xlsx") == dta_filename:
            print(f"Processing {dta['name']}")

            polygon_gdf = gpd.GeoDataFrame(geometry=[dta['geometry']])
            polygon_gdf.crs = "EPSG:4326"
        
            polygon_gdf_reprojected = polygon_gdf.to_crs(src.crs)
            
            clipped_bands = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            for i in range(1, 17):
                if i == 2 or i == 3 or i == 4 or i >= 13:
                    print("clipping band ke-" + str(i))
                    clipped, transformed = mask(bands_src[i], polygon_gdf_reprojected.geometry, crop=True)
                    clipped_bands[i] = clipped[0]
                    # print(clipped[0])
            
            clipped_meta = new_meta.copy()
            clipped_meta.update({
                "height": clipped.shape[1],  # New height
                "width": clipped.shape[2],   # New width
                "transform": transformed     # Updated transform
            })
            
        
    print("Done clipping bands")
    serapan_a = clipped_bands[13].copy()
    serapan_b = clipped_bands[14].copy()
    serapan_c = clipped_bands[15].copy()
    serapan_d = clipped_bands[16].copy()
    # hasil_b2 = clipped_bands[2].copy()
    # hasil_b3 = clipped_bands[3].copy()
    # hasil_b4 = clipped_bands[4].copy()

    # normalized_b2 = clipped_bands[2] / clipped_bands[2].max() * 255
    # normalized_b3 = clipped_bands[3] / clipped_bands[3].max() * 255
    # normalized_b4 = clipped_bands[4] / clipped_bands[4].max() * 255
    
    
    # MATPLOTLIB TRY EXPORT ----------------------------------------------------------------


    # export ke folder f"{parent_dir}/Images/"
    # beri nama file f"{dta_filename.split(".")[0]}_raw.(jpg/png/webp)"
    
    # rgb_image = np.dstack((normalized_b4, normalized_b3, normalized_b2)).astype(np.uint8)    
    # plt.figure(figsize=(20, 12))      
    # plt.imshow(rgb_image)
    
    #---------------------------------------------------
    
    pixel_count = hasil.shape[0]
    # latitudes = []
    # longitudes = []
    serapan = [serapan_a,serapan_b,serapan_c,serapan_d]
    
    for i in range(0, pixel_count):
        j = x_all[i]
        k = y_all[i]
        try:
            if lahan[i] == 'crop':
                # hasil_b2[j][k] = 255  # Bright Yellow
                # hasil_b3[j][k] = 255
                # hasil_b4[j][k] = 0
                serapan[0][j][k] = 62
                serapan[1][j][k] = 71
                serapan[2][j][k] = 88
                serapan[3][j][k] = 91
                # class_luas[lahan[i].title()] += 400
            elif lahan[i] == 'agriculture':
                # hasil_b2[j][k] = 255  # Light Orange
                # hasil_b3[j][k] = 165
                # hasil_b4[j][k] = 0
                serapan[0][j][k] = 45
                serapan[1][j][k] = 53
                serapan[2][j][k] = 67
                serapan[3][j][k] = 72
                # class_luas[lahan[i].title()] += 400
            elif lahan[i] == 'grassland':
                # hasil_b2[j][k] = 50   # Light Green
                # hasil_b3[j][k] = 205
                # hasil_b4[j][k] = 50
                serapan[0][j][k] = 39
                serapan[1][j][k] = 61
                serapan[2][j][k] = 74
                serapan[3][j][k] = 80
                # class_luas[lahan[i].title()] += 400
            elif lahan[i] == 'settlement':
                # hasil_b2[j][k] = 255  # Light Gray
                # hasil_b3[j][k] = 0
                # hasil_b4[j][k] = 0
                serapan[0][j][k] = 57
                serapan[1][j][k] = 72
                serapan[2][j][k] = 81
                serapan[3][j][k] = 86
                # class_luas[lahan[i].title()] += 400
            elif lahan[i] == 'tank_road_river':
                # hasil_b2[j][k] = 101  # Dark Brown
                # hasil_b3[j][k] = 67
                # hasil_b4[j][k] = 33
                serapan[0][j][k] = 98
                serapan[1][j][k] = 98
                serapan[2][j][k] = 98
                serapan[3][j][k] = 98
                # class_luas[lahan[i].title()] += 400
            elif lahan[i] == 'forest':
                # hasil_b2[j][k] = 34   # Dark Green
                # hasil_b3[j][k] = 139
                # hasil_b4[j][k] = 34
                serapan[0][j][k] = 25
                serapan[1][j][k] = 55
                serapan[2][j][k] = 70
                serapan[3][j][k] = 77
                # class_luas[lahan[i].title()] += 400
            elif lahan[i] == 'land_without_scrub':
                # hasil_b2[j][k] = 210  # Sandy Brown
                # hasil_b3[j][k] = 180
                # hasil_b4[j][k] = 140
                serapan[0][j][k] = 45
                serapan[1][j][k] = 66
                serapan[2][j][k] = 77
                serapan[3][j][k] = 83
                # class_luas[lahan[i].title()] += 400
            
            # projected_x, projected_y = xy(transformed, j, k)
            
            # transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
            # # lon, lat = transform(bands_src[i].crs, [projected_x], [projected_y])
            # lon, lat = transformer.transform(projected_x, projected_y)

            # # latitude, longitude = xy(transformed, j, k)
            # # print(lat)
            # # print(lon)
            # latitudes.append(lat)
            # longitudes.append(lon)
        except:
            ""
    # Menentukan path untuk menyimpan gambar
    output_list = ["A", "B", "C", "D"]

    for idx, alphabet in enumerate(output_list):
        output_filename = f"{dta_filename.split('.')[0]}_{alphabet}.jp2"  
        output_path = os.path.join(save_dir, output_filename)
        plt.figure(figsize=(10, 8))
        plt.imshow(serapan[idx], cmap="viridis")  # Choose a colormap (e.g., 'viridis', 'gray', etc.)
        plt.colorbar(label="Pixel Value")  # Add a color bar
        plt.title("2D Raster Image")
        plt.xlabel("Column Index")
        plt.ylabel("Row Index")
        plt.show()
        with rasterio.open(output_path, "w", **clipped_meta) as dst:
            dst.write(serapan[idx], 1)
    
    # print(len(latitudes))
    # print(len(longitudes))
    # output_filename = f"{dta_filename.split('.')[0]}_raw.png"  
    # output_path = os.path.join(save_dir, output_filename)
    
    # luas_filename = f"{dta_filename.split('.')[0]}_luas.csv"
    # output_luas = os.path.join(save_dir, luas_filename)
    
    # downloadable = f"{dta_filename.split('.')[0]}_download.csv"
    # output_dld = os.path.join(save_dir, downloadable)
    
    
    # # print(class_luas)
    # result_dld = pd.DataFrame({"Latitude":latitudes,
    #                             "Longitude": longitudes,
    #                             "A": serapan[0],
    #                             "B": serapan[1],
    #                             "C": serapan[2],
    #                             "D": serapan[3]
    #                             })

    # result_dld.to_csv(output_dld)
    
    # result_luas = pd.DataFrame(list(class_luas.items()), columns=['kelas', 'luas'])

    # result_luas.to_csv(output_luas)

    # # Menyimpan gambar
    # plt.savefig(output_path, format='png') 
    # print(f"Gambar berhasil disimpan di {output_path}")

# MATPLOTLIB TRY EXPORT ----------------------------------------------------------------
    # export ke folder f"{parent_dir}/Images/"
    # beri nama file f"{dta_filename.split(".")[0]}_classified.(jpg/png/webp)"

    # legend_patches = [
    #     mpatches.Patch(color=[255/255, 255/255, 0], label='Crop'),            # Bright Yellow
    #     mpatches.Patch(color=[255/255, 165/255, 0], label='Agriculture'),      # Light Orange
    #     mpatches.Patch(color=[50/255, 205/255, 50/255], label='Grassland'),    # Light Green
    #     mpatches.Patch(color=[255/255, 0/255, 0/255], label='Settlement'), # Light Gray
    #     mpatches.Patch(color=[101/255, 67/255, 33/255], label='Tank_road_river'), # Dark Brown
    #     mpatches.Patch(color=[34/255, 139/255, 34/255], label='Forest'),       # Dark Green
    #     mpatches.Patch(color=[210/255, 180/255, 140/255], label='Land Without Scrub'), # Sandy Brown
    # ]

    # rgb_image = np.dstack((hasil_b2, hasil_b3, hasil_b4)).astype(np.uint8)
    # plt.figure(figsize=(20, 12))  # Set width to 10 inches, height to 6 inches
    # plt.imshow(rgb_image)
    # plt.legend(handles=legend_patches, loc='upper left', fontsize='medium')
    
    # # Menentukan path untuk menyimpan gambar
    # output_filename = f"{dta_filename.split('.')[0]}_classified.png"  
    # output_path = os.path.join(save_dir, output_filename)

    # Menyimpan gambar
    # plt.savefig(output_path, format='png')
    # plt.show()
    # plt.close()  

    # print(f"Gambar berhasil disimpan di {output_path}")
# --------------------------------------------------------------------------

predict_all()