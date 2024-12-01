#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:51:39 2024

@author: MSI
"""

import os
import sys
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)


DTA = gpd.read_file(script_dir + "/mygeodata.zip")
real_names = []
names = []
for idx, dta in DTA.iterrows():
    name = dta['name'] + ".xlsx"
    name = name.replace("/", "_")
    name = name.replace(" ", "")
    real_names.append(dta['name'])
    names.append(name)

out_df = pd.DataFrame({'real_names': real_names,
                       'dta_filenames': names})

out_df.to_csv(script_dir + '/dta_filenames.csv')
