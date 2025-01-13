#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 12:51:23 2025

@author: MSI
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)
save_dir = os.path.join(parent_dir, 'Images')

import matplotlib.pyplot as plt
import rasterio

# Open the raster file
with rasterio.open("/Users/MSI/Development/ProDS1/ProDS2/Images/14DTACirasea_A.jp2") as src:
    raster_data = src.read(1)  # Read the first band (2D array)

# Plot the raster data
plt.figure(figsize=(10, 8))
plt.imshow(raster_data, cmap="viridis")  # Choose a colormap (e.g., 'viridis', 'gray', etc.)
plt.colorbar(label="Pixel Value")  # Add a color bar
plt.title("2D Raster Image")
plt.xlabel("Column Index")
plt.ylabel("Row Index")
plt.show()