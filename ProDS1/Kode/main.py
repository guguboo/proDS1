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
from shapely import Polygon
import geopandas as gpd
import rasterio.crs as CRS
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np

script_directory = os.path.dirname(os.path.abspath(__file__))
# %% dataset load resolusi 10m (download dulu di drive, lalu di folder jp2 buat folder 10m dan extract datanya di situ)

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


print(B1_20)
print(B1_20.shape)

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

# %%  buat dataset labelled (latihan) 10m resolusi

# geojson_path = script_directory + "/geojson/"

# geojson_filename = ["labelling_latihan_1.geojson", "labelling_latihan_2.geojson"]


# B2_output = []
# B3_output = []
# B4_output = []
# labels_output = []

# label = ""

# for file in geojson_filename: 
#     out_of_bound_count = 0
#     print("proses file " + file)
#     multipoints_gdf = gpd.read_file(geojson_path + file)
#     multipoints_gdf = multipoints_gdf.to_crs(b2_src.crs)
    
#     for index, kategori in multipoints_gdf.iterrows():
#         multipoint_geometry = kategori['geometry']
        
#         if index == 0:
#             label = "bangunan"
#         elif index == 1:
#             label = "area_hijau"
#         else:
#             label = "air"
            
#         for point in multipoint_geometry.geoms:
#             x, y = point.x, point.y
#             x_raster, y_raster = b2_src.index(x, y)
            
            
#             try:
#                 B2_output.append(B2[x_raster][y_raster])
#                 B3_output.append(B3[x_raster][y_raster])
#                 B4_output.append(B4[x_raster][y_raster])
#                 labels_output.append(label)
#             except:
#                 out_of_bound_count += 1
                
                
#     print("koordinat2 yang out of bound :" + str(out_of_bound_count))
# #output file dalam excel
# output_counter = 1
# done_output = False
# output_filename = 'dataset_satelit_latihan'
# out_df = pd.DataFrame({'B2': B2_output, 'B3': B3_output, 'B4': B4_output, 'jenis_lahan': labels_output})

# while not done_output:
#     try:    
#         out_df.to_excel(script_directory + '/output_labelling/' + output_filename + "_" + str(output_counter) + ".xlsx", index=False)
#         done_output = True
#     except:
#         output_counter += 1

#%% buat dataset labelled untuk resolusi 20m
geojson_path = script_directory + "/geojson/"

geojson_filename = ["labelling_latihan_1.geojson", "labelling_latihan_2.geojson", "labelling_latihan_3.geojson"]
jumlah_labeled_file = 4

B1_output = []
B2_output = []
B3_output = []
B4_output = []
B5_output = []
B6_output = []
B7_output = []
B8A_output = []
B11_output = []
B12_output = []
labels_output = []

label = ""

for file_number in range(1, jumlah_labeled_file+1): 
    out_of_bound_count = 0
    filename = "labelling_latihan_" + str(file_number) + ".geojson"
    print("proses file " + filename)
    multipoints_gdf = gpd.read_file(geojson_path + filename)
    multipoints_gdf = multipoints_gdf.to_crs(b1_src_20.crs)
    
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
            x_raster, y_raster = b1_src_20.index(x, y)
            
            b1_around = 0
            b2_around = 0
            b3_around = 0
            b4_around = 0
            b5_around = 0
            b6_around = 0
            b7_around = 0
            b8_around = 0
            b11_around = 0
            b12_around = 0
            
            cnt = 0
            for i in range(x_raster-1, x_raster+2):
                for j in range(y_raster-1, y_raster+2):
                    try:
                        b2_around += B2_20[i][j]
                        b1_around += B1_20[i][j]
                        b3_around += B3_20[i][j]
                        b4_around += B4_20[i][j]
                        b5_around += B5_20[i][j]
                        b6_around += B6_20[i][j]
                        b7_around += B7_20[i][j]
                        b8_around += B8A_20[i][j]
                        b11_around += B11_20[i][j]
                        b12_around += B12_20[i][j]
                        
                        cnt += 1
                    except:
                        ""
            try:
                B1_output.append(b1_around / cnt)
                B2_output.append(b2_around / cnt)
                B3_output.append(b3_around / cnt)
                B4_output.append(b4_around / cnt)
                B5_output.append(b5_around / cnt)
                B6_output.append(b6_around / cnt)
                B7_output.append(b7_around / cnt)
                B8A_output.append(b8_around / cnt)
                B11_output.append(b11_around / cnt)
                B12_output.append(b12_around / cnt)
                labels_output.append(label)
            except:
                out_of_bound_count += 1
                                
    print("koordinat2 yang out of bound :" + str(out_of_bound_count))
    print()
    

#%% output ke excel


output_counter = 1
done_output = False
output_filename = 'coba_coba_20m'
out_df = pd.DataFrame({'B1': B1_output, 'B2': B2_output, 'B3': B3_output, 'B4': B4_output, 'B5': B5_output, 'B6': B6_output, 'B7': B7_output, 'B8': B8A_output, 'B11': B11_output, 'B12': B12_output, 'jenis_lahan': labels_output})

out_df = out_df.drop_duplicates()


out_df.to_excel(script_directory + '/output_labelling/' + output_filename + "_" + str(output_counter) + ".xlsx", index=False)
done_output = True



# %% function untuk meng upscale resolusi yang 20m

def upscale_array(a_array):
    height, width = a_array.shape

    upscaled_array = np.zeros((height*2, width*2), dtype=a_array.dtype)

    for i in range(height*2):
        for j in range(width*2):
            orig_i, orig_j = i // 2, j // 2
            upscaled_array[i, j] = a_array[orig_i, orig_j]

    return upscaled_array

# %% try geopandas
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

#%% pembuatan dataset clipped
output_arr = [[],[],[],[],[],[],[],[],[],[],[],[]]
done_xy = False

for band_idx in range(0, 10):
    clip = arr_of_clipped[band_idx][0]
    for row_idx in range(0,len(clip)):
        clip_row = clip[row_idx]
        for col_idx in range(0, len(clip_row)):
            item = clip_row[col_idx]
            if item != 0:                
                output_arr[band_idx].append(item)
                if not done_xy:
                    output_arr[10].append(row_idx)
                    output_arr[11].append(col_idx)
                    
    done_xy = True
                
    print(f"Band ke-{band_idx + 1} beres diclip")
print("sudah selesai") 

#%% coba liat outputnya    
output_filename = 'dataset_coba_prediksi'
out_df = pd.DataFrame({'B1': output_arr[0], 'B2': output_arr[1], 'B3': output_arr[2], 'B4': output_arr[3], 'B5': output_arr[4], 'B6': output_arr[5], 'B7': output_arr[6], 'B8': output_arr[7], 'B11': output_arr[8], 'B12': output_arr[9], 'x': output_arr[10], 'y': output_arr[11]})

out_df.to_excel(script_directory + '/coba_remapping/' + output_filename + ".xlsx", index=False)
# %% show rgb

normalized_b2 = clipped_b2[0] / clipped_b2[0].max() * 255
normalized_b3 = clipped_b3[0] / clipped_b3[0].max() * 255
normalized_b4 = clipped_b4[0] / clipped_b4[0].max() * 255


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


#%% PETAKAN KEMBALI KE 2d

hasil_prediksi = pd.read_excel(script_directory +"/prediction_result/real_predict/prediction_result.xlsx")

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
    if lahan[i] == 'bangunan':
        hasil_b2[x][y] = 200
        hasil_b3[x][y] = 0
        hasil_b4[x][y] = 0
    elif lahan[i] == 'area_hijau':
        hasil_b2[x][y] = 0
        hasil_b3[x][y] = 200
        hasil_b4[x][y] = 0
    else:
        hasil_b2[x][y] = 0
        hasil_b3[x][y] = 0
        hasil_b4[x][y] = 200
        

rgb_image = np.dstack((hasil_b2, hasil_b3, hasil_b4)).astype(np.uint8)
plt.figure(figsize=(12, 12))  # Set width to 10 inches, height to 6 inches
plt.imshow(rgb_image)
        

