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
import os
import sys
import pandas as pd
import geopandas as gpd

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)
import addFeature as af
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
n_vico = 1
class_vico = ["forest", "land_without_scrub"]

label_mark = parent_dir + "/Labeling/mark_"
n_mark = 1
class_mark = ["river", "tank"]

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
        
        for index, kategori in multipoints_gdf.iterrows():
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
output_filename = 'ProDS2'

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
    
    })


out_df['NDVI'] = af.addNDVI(out_df['B4'], out_df['B8'])
out_df['EVI'] = af.addEVI(out_df['B2'], out_df['B4'], out_df['B8'])
out_df['land_cover'] = label_output

out_df = out_df.drop_duplicates()

print(out_df.shape)

out_df.to_excel(parent_dir + '/Labeled/labeling_by_pixel_' + output_filename + ".xlsx", index=False)

