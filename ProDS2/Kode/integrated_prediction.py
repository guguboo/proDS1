#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:33:36 2024

@author: MSI
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 13:07:49 2024

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
import numpy as np

from sklearn.ensemble import RandomForestClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)

import time

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

def predict_all():
    for i in range(59):
        prediction(i)

def prediction(dta_idx):
    start_time = time.time()
    with open(script_dir + '/last_fetched.txt', 'r') as file:
        date_filename = file.read().strip()

    dta_df = pd.read_csv(script_dir + "/dta_filenames.csv")
    dta_filename = dta_df['dta_filenames'][dta_idx]
        
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
    
    predict_data = pd.read_excel(parent_dir + '/Data/(to predict)/' + dta_filename)

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
    features_stage_2 = ['B3', 'B4', 'B5', 'B8', 'B11']
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


    bands_src = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    band_path = parent_dir + '/Data/satelit/21082024/'

    for i in range(1, 13):
        if i == 2 or i == 3 or i == 4:
            curr_band = rasterio.open(band_path + "B" + str(i) + ".jp2")
            bands_src[i] = curr_band

    print("Successfully Read Bands Raster")
    
    DTA = gpd.read_file(script_dir + "/mygeodata.zip")
    
    src = bands_src[2]
    for idx, dta in DTA.iterrows():
        name = dta['name'] + ".xlsx"
        name = name.replace("/", "_")
        name = name.replace(" ", "")
        if name == dta_filename:
            print(f"Processing {dta['name']}")
            polygon_gdf = gpd.GeoDataFrame(geometry=[dta['geometry']])
            polygon_gdf.crs = "EPSG:4326"
        
            polygon_gdf_reprojected = polygon_gdf.to_crs(src.crs)
            
            clipped_bands = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            
            for i in range(1, 13):
                if i == 2 or i == 3 or i == 4:
                    print("clipping band ke-" + str(i))
                    clipped, transformed = mask(bands_src[i], polygon_gdf_reprojected.geometry, crop=True)
                    clipped_bands[i] = clipped[0]
            
            
        
    print("Done clipping bands")
    hasil_b2 = clipped_bands[2].copy()
    hasil_b3 = clipped_bands[3].copy()
    hasil_b4 = clipped_bands[4].copy()

    normalized_b2 = clipped_bands[2] / clipped_bands[2].max() * 255
    normalized_b3 = clipped_bands[3] / clipped_bands[3].max() * 255
    normalized_b4 = clipped_bands[4] / clipped_bands[4].max() * 255
    
    
    # MATPLOTLIB TRY EXPORT ----------------------------------------------------------------
    # export ke folder f"{parent_dir}/Images/"
    # beri nama file f"{dta_filename.split(".")[0]}_raw.(jpg/png/webp)"
    
    rgb_image = np.dstack((normalized_b4, normalized_b3, normalized_b2)).astype(np.uint8)    
    plt.figure(figsize=(20, 12))      
    plt.imshow(rgb_image)
    
    #---------------------------------------------------
    
    pixel_count = hasil.shape[0]
    for i in range(0, pixel_count):
        j = x_all[i]
        k = y_all[i]
        try:
            if lahan[i] == 'crop':
                hasil_b2[j][k] = 255  # Bright Yellow
                hasil_b3[j][k] = 255
                hasil_b4[j][k] = 0
            elif lahan[i] == 'agriculture':
                hasil_b2[j][k] = 255  # Light Orange
                hasil_b3[j][k] = 165
                hasil_b4[j][k] = 0
            elif lahan[i] == 'grassland':
                hasil_b2[j][k] = 50   # Light Green
                hasil_b3[j][k] = 205
                hasil_b4[j][k] = 50
            elif lahan[i] == 'settlement':
                hasil_b2[j][k] = 255  # Light Gray
                hasil_b3[j][k] = 0
                hasil_b4[j][k] = 0
            elif lahan[i] == 'tank_road_river':
                hasil_b2[j][k] = 101  # Dark Brown
                hasil_b3[j][k] = 67
                hasil_b4[j][k] = 33
            elif lahan[i] == 'forest':
                hasil_b2[j][k] = 34   # Dark Green
                hasil_b3[j][k] = 139
                hasil_b4[j][k] = 34
            elif lahan[i] == 'land_without_scrub':
                hasil_b2[j][k] = 210  # Sandy Brown
                hasil_b3[j][k] = 180
                hasil_b4[j][k] = 140
        except:
            ""
    # Menentukan path untuk menyimpan gambar
    output_filename = f"{dta_filename.split('.')[0]}_raw.png"  
    output_path = os.path.join(save_dir, output_filename)

    # Menyimpan gambar
    plt.savefig(output_path, format='png') 
    print(f"Gambar berhasil disimpan di {output_path}")

# MATPLOTLIB TRY EXPORT ----------------------------------------------------------------
    # export ke folder f"{parent_dir}/Images/"
    # beri nama file f"{dta_filename.split(".")[0]}_classified.(jpg/png/webp)"

    legend_patches = [
        mpatches.Patch(color=[255/255, 255/255, 0], label='Crop'),            # Bright Yellow
        mpatches.Patch(color=[255/255, 165/255, 0], label='Agriculture'),      # Light Orange
        mpatches.Patch(color=[50/255, 205/255, 50/255], label='Grassland'),    # Light Green
        mpatches.Patch(color=[255/255, 0/255, 0/255], label='Settlement'), # Light Gray
        mpatches.Patch(color=[101/255, 67/255, 33/255], label='Tank_road_river'), # Dark Brown
        mpatches.Patch(color=[34/255, 139/255, 34/255], label='Forest'),       # Dark Green
        mpatches.Patch(color=[210/255, 180/255, 140/255], label='Land Without Scrub'), # Sandy Brown
    ]

    rgb_image = np.dstack((hasil_b2, hasil_b3, hasil_b4)).astype(np.uint8)
    plt.figure(figsize=(20, 12))  # Set width to 10 inches, height to 6 inches
    plt.imshow(rgb_image)
    plt.legend(handles=legend_patches, loc='upper left', fontsize='medium')
    
    # Menentukan path untuk menyimpan gambar
    output_filename = f"{dta_filename.split('.')[0]}_classified.png"  
    output_path = os.path.join(save_dir, output_filename)

    # Menyimpan gambar
    plt.savefig(output_path, format='png')
    # plt.show()
    plt.close()  

    print(f"Gambar berhasil disimpan di {output_path}")
# --------------------------------------------------------------------------

predict_all()