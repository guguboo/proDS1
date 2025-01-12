#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 08:00:50 2024

@author: MSI
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 10:53:20 2024

@author: MSI
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 11:01:34 2024

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

#%%
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
#%% CLIPPING

# print(test_geojson[0]["coordinates"][0])
# polygon_gdf = gpd.read_file(geojson_path + file) 
# polygon_gdf = gpd.GeoDataFrame(geometry=[Polygon(my_geojson[0]["coordinates"])])

DTA_Cisangkuy = gpd.read_file(script_dir + "/dta_cisangkuy.geojson")

src = bands_src[1]
polygon_gdf = DTA_Cisangkuy
polygon_gdf.crs = "EPSG:4326"
polygon_gdf_reprojected = polygon_gdf.to_crs(bands_src[1].crs)

clipped_bands = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(1, 13):
    if i != 9 and i != 10:
        clipped, transformed = mask(bands_src[i], polygon_gdf_reprojected.geometry, crop=True)
        clipped_bands[i] = clipped[0]

#%% pembuatan dataset clipping yang belum dilabel (menghiraukan 0)

output_arr = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
done_xy = False

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
                    output_arr[i].append(item)
                    if not done_xy:
                        output_arr[13].append(row_idx)
                        output_arr[14].append(col_idx)
                    
        done_xy = True

print("Done clipping bands")
#%% import df
with open(script_dir + '/filename.txt', 'r') as file:
    content = file.read().strip()
filename = content
dta = "05DTACiranjeng_CisangkuyHulu_pixel.xlsx"
labeled = parent_dir + "/Labeled/labeling_by_pixel_"
training = pd.read_excel(labeled + filename)
predict_data = pd.read_excel(parent_dir + '/Data/(to predict)/' + dta)
# print(predict_data.shape)

#%% TAHAP PERTAMA

start_time = time.time()
features_stage_1 = ["B1", "B5", "B11", "B12"]

labeled = parent_dir + "/Labeled/labeling_by_pixel_"
filename = content
df = pd.read_excel(labeled+filename)

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
hasil_1 = rfc_1.predict(predict_data[features_stage_1])

print(len(hasil_1)) 

print(sum(hasil_1=="group1"))
df_g1 = predict_data.iloc[hasil_1=="group1"]
df_g2 = predict_data.iloc[hasil_1=="group2"]
df_g3 = predict_data.iloc[hasil_1=="group3"]


# print(df_g1['land_cover'].value_counts(),end="\n\n")
# print(df_g2['land_cover'].value_counts(),end="\n\n")
# print(df_g3['land_cover'].value_counts(),end="\n\n")

end_time = time.time()

print(f"Time taken to predict: {end_time - start_time:.2f} seconds")
#%% TAHAP 2 GROUP 1

features_stage_2 = ['B3', 'B4', 'B5', 'B8', 'B11']
X = df[df['land_cover'].isin(group_1)]
y = X['land_cover']

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

res_group_1

#%% TAHAP 2 GROUP 3

features_stage_3 = ['B1', 'B2', 'B4', 'B5', 'B11', 'B12']
X = df[df['land_cover'].isin(group_3)]
y = X['land_cover']

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

res_group_3


#%% TAHAP 2 GROUP 2
features_stage_2 = ['B5', 'B7', 'B8', 'B12']
X = df[df['land_cover'].isin(group_2)]
y = X['land_cover']

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

res_group_2

#%% concat

hasil = pd.concat([res_group_1, res_group_2, res_group_3], axis=0)
hasil.head()

lahan = hasil['prediction']
x_all = hasil['x']
y_all = hasil['y']

hasil_b2 = clipped_bands[2].copy()
hasil_b3 = clipped_bands[3].copy()
hasil_b4 = clipped_bands[4].copy()


pixel_count = hasil.shape[0]
for i in range(0, pixel_count):
    j = x_all[i]
    k = y_all[i]
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

plt.show()

# Define legend patches for each land cover type

