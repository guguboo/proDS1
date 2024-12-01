from flask import Flask, render_template, jsonify
# Geospatial processing packages
import geopandas as gpd
import geojson

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
ee.Initialize(project='vics-testing-gee')

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

@app.route('/')
def index():
    ## Map
    # Load the shapefile into a GeoDataFrame
    citarum_gdf = gpd.read_file(script_dir + "/mygeodata.zip")

    print(citarum_gdf.columns)
    print(citarum_gdf.head())


    # Create a GeoDataFrame for the region
    region_geom = citarum_gdf.unary_union  # Union all geometries into one
    region_geojson = mapping(region_geom)   # Convert shapely geometry to GeoJSON
    region_ee = ee.Geometry(region_geojson) # Convert GeoJSON to Earth Engine geometry

    # Generate RGB image using GEE
    image = generate_image(
        region_ee,
        product='COPERNICUS/S2_SR_HARMONIZED', # Sentinel-2A
        min_date='2023-06-01', # Get all images within
        max_date='2024-12-31', # the year 2021
        cloud_pct=30, # Filter out images with cloud cover >= 30.0%
    )

    # Visualize 
    region_centroid = region_geom.centroid
    region_centroid_x = region_centroid.x
    region_centroid_y = region_centroid.y
    Map = emap.Map(center=[region_centroid_y, region_centroid_x], zoom=10)

    Map.addLayer(image, {}, 'Sentinel2')

    # Convert the GeoDataFrame geometries to Earth Engine geometries
    regions = []
    features = []
    for _, row in citarum_gdf.iterrows():
        # Convert each geometry to Earth Engine format
        region = ee.Geometry(mapping(row['geometry']))
        name = row['name']
        feature = ee.Feature(region).set('name', name)
        regions.append(region)
        features.append(feature)

    # Merge into a feature collection and style based on 'name'
    fc = ee.FeatureCollection(features)
    Map.addLayer(fc, {'color': 'red', 'width': 1, 'fillColor': '00000000'}, 'DTA Citarum')

    Map.addLayerControl()
    map_html = Map._parent.get_root().render()

    ## Nama DTA
    names = citarum_gdf['name'].tolist()

    ## Area
    citarum_gdf = citarum_gdf.to_crs(epsg=3395)
    citarum_gdf['area'] = citarum_gdf.geometry.area
    citarum_gdf['area'] = citarum_gdf['area'].astype(float)/1000000
    citarum_gdf['area'] = citarum_gdf['area'].apply(lambda x: f"{x:.2f}")
    area_dict = citarum_gdf.set_index('name')['area'].to_dict()

    m = geemap.Map()

    # Add GeoJSON data with styling
    m.add_data(
        citarum_gdf,
        column='area',            # Column to style by (e.g., 'area')
        cmap='Blues',             # Colormap
        legend_title='Area',      # Legend title
    )

    map_html = m._parent.get_root().render()

    return render_template('index.html', map_html=map_html, dta=names, area=area_dict)

@app.route('/dta_geojson')
def dta_geojson():
    citarum_gdf = gpd.read_file(script_dir + "/mygeodata.zip")
    citarum_gdf = citarum_gdf.to_crs(epsg=4326)  # Pastikan CRS dalam WGS84 (longitude, latitude)

    # Tambahkan properti 'name' untuk setiap feature
    citarum_gdf['name'] = citarum_gdf['name']
    geojson_data = citarum_gdf.to_json()

    return jsonify(geojson.loads(geojson_data))

if __name__ == '__main__':
    print("Hello")
    app.run(debug=True, threaded = False)
