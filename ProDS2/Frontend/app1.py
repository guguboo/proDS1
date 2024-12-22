from flask import Flask, render_template, jsonify, request
import pandas as pd
# Geospatial processing packages
import geopandas as gpd
import geojson
import json
import re
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)

import shapely
import rasterio as rio
from rasterio.plot import show
import rasterio.mask
from shapely.geometry import box
from shapely.geometry import mapping

# Mapping and plotting libraries
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import ee
import eeconvert as eec
import geemap
import geemap.eefolium as emap
import folium
import mapclassify

from shapely.geometry import Polygon

import asyncio

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


app = Flask(__name__)

print(ee.__version__)

# Melakukan Autentikasi Pengguna. Klik link yang muncul, kemudian copykan
# token yang diperoleh ke kolom yang tersedia
ee.Authenticate()
print("testing, apakah berjalan")

# Mengaktifkan GEE pada Google Colab
ee.Initialize(project='ee-atlk')

citarum_gdf = ""

def generate_image(
    region,
    product='COPERNICUS/S2',
    min_date='2018-01-01',
    max_date='2020-01-01',
    range_min=0,
    range_max=2000,
    cloud_pct=10
):

    """Generates cloud-filtered, median-aggregated
    Sentinel-2 image from Google Earth Engine using the
    Pythin Earth Engine API.

    Args:
      region (ee.Geometry): The geometry of the area of interest to filter to.
      product (str): Earth Engine asset ID
        You can find the full list of ImageCollection IDs
        at https://developers.google.com/earth-engine/datasets
      min_date (str): Minimum date to acquire collection of satellite images
      max_date (str): Maximum date to acquire collection of satellite images
      range_min (int): Minimum value for visalization range
      range_max (int): Maximum value for visualization range
      cloud_pct (float): The cloud cover percent to filter by (default 10)

    Returns:
      ee.image.Image: Generated Sentinel-2 image clipped to the region of interest
    """

    # Generate median aggregated composite
    image = ee.ImageCollection(product)\
        .filterBounds(region)\
        .filterDate(str(min_date), str(max_date))\
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))\
        .median()

    # Get RGB bands
    image = image.visualize(bands=['B4', 'B3', 'B2'], min=range_min, max=range_max)
    # Note that the max value of the RGB bands is set to 65535
    # because the bands of Sentinel-2 are 16-bit integers
    # with a full numerical range of [0, 65535] (max is 2^16 - 1);
    # however, the actual values are much smaller than the max value.
    # Source: https://stackoverflow.com/a/63912278/4777141

    return image.clip(region)

def display_map(df):
    return ''

@app.route('/')
def index():
    ## Map
    # Load the shapefile into a GeoDataFrame
    global citarum_gdf
    citarum_gdf = gpd.read_file(script_dir + "/mygeodata.zip")

    print(citarum_gdf.columns)
    print(citarum_gdf.head())

    citarum_gdf.loc[15, 'id'] = 'ID_00058'
    citarum_gdf.loc[16, 'id'] = 'ID_00059'
    citarum_gdf.loc[44, 'id'] = 'ID_00060'

    ids = citarum_gdf['id'].tolist()

    ## Nama DTA
    names = citarum_gdf.set_index('id')['name'].to_dict()
    
    ## Area
    citarum_gdf = citarum_gdf.to_crs(epsg=3395)
    citarum_gdf['area'] = citarum_gdf.geometry.area
    citarum_gdf['area'] = citarum_gdf['area'].astype(float)/1000000
    citarum_gdf['area'] = citarum_gdf['area'].apply(lambda x: f"{x:.2f}")
    area_dict = citarum_gdf.set_index('id')['area'].to_dict()

    return render_template('index copy.html', id=ids, dta=names, area=area_dict)

@app.route('/select', methods=['POST'])
def selected():
    global citarum_gdf
    data = request.get_json()  # Mendapatkan data JSON dari frontend
    selected_id = data.get('id')  # Mengambil ID dari permintaan
    print(f"ID yang diterima: {selected_id}")  # Logging untuk debug
    selected_region = citarum_gdf[citarum_gdf['id'] == selected_id]

    selected_name = selected_region['name'].iloc[0]
    selected_name = re.sub(r"[^a-zA-Z0-9\s/]", '', selected_name)  # Tetap menghapus karakter khusus, kecuali '/'
    selected_name = re.sub(r"/", '_', selected_name)  # Ganti '/' dengan '_'
    selected_name = re.sub(r"\s+", '', selected_name)  # Hapus semua spasi

    file_name = selected_name + '_luas.csv'

    try:
        # Menggabungkan path dengan os.path.join
        file_path = os.path.join(script_dir, "../Images", file_name)
        selected_details = pd.read_csv(file_path)
        print(selected_details)
        
        # Konversi DataFrame menjadi JSON
        selected_details_json = selected_details.to_dict(orient='records')
        response_data = {
            "message": f"ID {selected_id} telah diproses.",
            "details": selected_details_json
        }
        return jsonify(response_data)
    except FileNotFoundError as e:
        print(f"File not found: {e}")

    # Lakukan sesuatu dengan ID (misalnya, query ke database)
    # response_message bisa diubah sesuai kebutuhan
    response_message = f"ID {selected_id} telah diproses."

    return jsonify({"message": response_message})

if __name__ == '__main__':
    print("Hello")
    app.run(debug=True)
