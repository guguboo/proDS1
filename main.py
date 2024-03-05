#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:39:34 2024

@author: MSI
"""

# %% importing libraries
import rasterio
import geopandas as gpd
from osgeo import gdal, osr
import rasterio.crs as CRS
from rasterio.mask import mask
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import shape, Polygon
import pyproj
from fiona.crs import from_epsg

# %% dataset load
b2_path = '/Users/MSI/Development/ProDS1/jp2/10m/T48MYT_20231220T030131_B02_10m.jp2'
b3_path = '/Users/MSI/Development/ProDS1/jp2/10m/T48MYT_20231220T030131_B03_10m.jp2'
b4_path = '/Users/MSI/Development/ProDS1/jp2/10m/T48MYT_20231220T030131_B04_10m.jp2'

b2_src = rasterio.open(b2_path)
b3_src = rasterio.open(b3_path)
b4_src = rasterio.open(b4_path)

B2 = b2_src.read(1)
B3 = b3_src.read(1)
B4 = b4_src.read(1)

# %% METADATA


print(b2_src.meta)

# %% geoJSON

xmin, ymin, xmax, ymax = b2_src.bounds

x_range = xmax - xmin
y_range = ymax - ymin

xmin = xmin + (x_range / 3)
ymin = ymin + (y_range / 3)
xmax = xmax - (x_range / 3)
ymax = ymax - (y_range / 3)

test_geojson = [{
    "type": "Polygon",
    "coordinates": [
            [
                [xmin, ymin],
                [xmax, ymin],
                [xmax, ymax],
                [xmin, ymax],
                [xmin, ymin]
            ]
        ]
    }]

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


# %% rgb
# my_image = np.dstack((B2/74.54, B3/70.7, B4/68.14))
# plt.imshow(my_image)

# %% try geopandas
print(my_geojson[0]["coordinates"][0])

polygon = Polygon(my_geojson[0]["coordinates"][0])
polygon_gdf = gpd.GeoDataFrame(geometry=[polygon])
polygon_gdf.crs = "EPSG:4326"  # Assuming WGS84
# polygon_gdf = gpd.GeoDataFrame(geometry=[Polygon(my_geojson[0]["coordinates"])])

# %% try clipping

polygon_gdf_reprojected = polygon_gdf.to_crs(b2_src.crs)

clipped_b2, transform_b2 = mask(b2_src, polygon_gdf_reprojected.geometry, crop=True)      

# %% clipping b3 & b4
clipped_b3, transform_b3 = mask(b3_src, polygon_gdf_reprojected.geometry, crop=True)
clipped_b4, transform_b4 = mask(b4_src, polygon_gdf_reprojected.geometry, crop=True)

# %% testing print
print(clipped_b2.max())
print(clipped_b3.max())
print(clipped_b4.max())

print(clipped_b2)

# %% show rgb

normalized_b2 = clipped_b2[0] / clipped_b2[0].max() * 600
normalized_b3 = clipped_b3[0] / clipped_b3[0].max() * 600
normalized_b4 = clipped_b4[0] / clipped_b4[0].max() * 600

print(normalized_b2.max())

rgb_image = np.dstack((normalized_b4, normalized_b3, normalized_b2)).astype(np.uint8)
rgb_raw = np.dstack((clipped_b2[0], clipped_b3[0], clipped_b4[0]))

plt.figure(figsize=(20, 12))  # Set width to 10 inches, height to 6 inches
plt.imshow(rgb_image)
# plt.imshow(rgb_raw)
