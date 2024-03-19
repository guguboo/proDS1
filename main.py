#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:39:34 2024

@author: MSI
"""

# %% importing libraries
import rasterio
import os
import pandas as pd
import geopandas as gpd
import rasterio.crs as CRS
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np

# %% dataset load resolusi 10m

script_directory = os.path.dirname(os.path.abspath(__file__))
print(script_directory)

b2_path = script_directory + '/jp2/10m/T48MYT_20231220T030131_B02_10m.jp2'
b3_path = script_directory + '/jp2/10m/T48MYT_20231220T030131_B03_10m.jp2'
b4_path = script_directory + '/jp2/10m/T48MYT_20231220T030131_B04_10m.jp2'

b2_src = rasterio.open(b2_path)
b3_src = rasterio.open(b3_path)
b4_src = rasterio.open(b4_path)

B2 = b2_src.read(1)
B3 = b3_src.read(1)
B4 = b4_src.read(1)

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

# %% METADATA


print(b7_src_20.meta)
# %% geoJSON

my_geojson = [{
    "type": "Polygon",
    "coordinates": [
        [
            [107.444916, -7.013668],
            [107.503967, -6.929153],
            [107.503967, -6.873256],
            [107.605591, -6.761443],
            [107.631683, -6.762806],
            [107.727814, -6.822807],
            [107.729187, -6.856895],
            [107.784284, -6.925051],
            [107.716827, -7.137686],
            [107.681059, -7.234919],
            [107.628398, -7.245696],
            [107.58566, -7.197292],
            [107.547941, -7.202733],
            [107.490234, -7.164938],
            [107.391357, -7.137686],
            [107.383118, -7.080451],
            [107.444916, -7.013668]
        ]
    ]
}]

bandung_geojson = [{
    "type": "Polygon",
    "coordinates": [
          [
            [
              107.64052855847251,
              -6.978469526805924
            ],
            [
              107.63956349477291,
              -6.979924135284435
            ],
            [
              107.64272675912213,
              -6.981520651582898
            ],
            [
              107.64354885042252,
              -6.979480657568772
            ],
            [
              107.6405285 5847251,
              -6.978469526805924
            ]
          ]
        ],
    }]

# %%  buat dataset labelled (latihan)

geojson_path = script_directory + "/geojson/"

geojson_filename = ["labelling_latihan_1.geojson", "labelling_latihan_2.geojson"]

B2_output = []
B3_output = []
B4_output = []
labels_output = []

label = ""

for file in geojson_filename: 
    out_of_bound_count = 0
    print("proses file " + file)
    multipoints_gdf = gpd.read_file(geojson_path + file)
    multipoints_gdf = multipoints_gdf.to_crs(b2_src.crs)
    
    for index, kategori in multipoints_gdf.iterrows():
        multipoint_geometry = kategori['geometry']
        
        if index == 0:
            label = "bangunan"
        elif index == 1:
            label = "area_hijau"
        else:
            label = "air"
            
        for point in multipoint_geometry.geoms:
            x, y = point.x, point.y
            x_raster, y_raster = b2_src.index(x, y)
            
            
            try:
                B2_output.append(B2[x_raster][y_raster])
                B3_output.append(B3[x_raster][y_raster])
                B4_output.append(B4[x_raster][y_raster])
                labels_output.append(label)
            except:
                out_of_bound_count += 1
                
                
    print("koordinat2 yang out of bound :" + str(out_of_bound_count))
#output file dalam excel
output_counter = 1
done_output = False
output_filename = 'dataset_satelit_latihan'
out_df = pd.DataFrame({'B2': B2_output, 'B3': B3_output, 'B4': B4_output, 'jenis_lahan': labels_output})

while not done_output:
    try:    
        out_df.to_excel(script_directory + '/output_labelling/' + output_filename + "_" + str(output_counter) + ".xlsx", index=False)
        done_output = True
    except:
        output_counter += 1
        
# %% try geopandas
print(my_geojson[0]["coordinates"][0])

polygon = Polygon(my_geojson[0]["coordinates"][0])
#polygon_gdf = gpd.GeoDataFrame(geometry=[polygon])
polygon_gdf = gpd.read_file(geojson_path + file) 
polygon_gdf.crs = "EPSG:4326"  # Assuming WGS84
# polygon_gdf = gpd.GeoDataFrame(geometry=[Polygon(my_geojson[0]["coordinates"])])

# %% try clipping

polygon_gdf_reprojected = polygon_gdf.to_crs(b2_src.crs)

print(polygon_gdf_reprojected)

clipped_b2, transform_b2 = mask(b2_src, polygon_gdf_reprojected.geometry, crop=True)      

# %% clipping b3 & b4
clipped_b3, transform_b3 = mask(b3_src, polygon_gdf_reprojected.geometry, crop=True)
clipped_b4, transform_b4 = mask(b4_src, polygon_gdf_reprojected.geometry, crop=True)

# %% testing print
print(clipped_b2.max())
print(clipped_b3.max())
print(clipped_b4.max())

print(clipped_b2.shape)

# %% show rgb

normalized_b2 = clipped_b2[0] / clipped_b2[0].max() * 500
normalized_b3 = clipped_b3[0] / clipped_b3[0].max() * 500
normalized_b4 = clipped_b4[0] / clipped_b4[0].max() * 500

print(normalized_b2.max())

rgb_image = np.dstack((normalized_b4, normalized_b3, normalized_b2)).astype(np.uint8)
rgb_raw = np.dstack((clipped_b2[0], clipped_b3[0], clipped_b4[0]))

plt.figure(figsize=(20, 12))  # Set width to 10 inches, height to 6 inches
plt.imshow(rgb_image)


# %% convert matrix 2d menjadi 1d
b2_flatten = normalized_b2.flatten()
b3_flatten = normalized_b3.flatten()
b4_flatten = normalized_b4.flatten()

print(b2_flatten)

out_df = pd.DataFrame({'B2': b2_flatten, 'B3': b3_flatten, 'B4': b4_flatten})

out_df.to_excel('testing_output.xlsx', index=False)

