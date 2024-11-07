from flask import Flask, render_template, jsonify
import geopandas as gpd

app = Flask(__name__)

@app.route('/')
def index():
    try:
        # Load the shapefile using GeoPandas
        shapefile_path = 'mygeodata/doc-polygon.shp'
        gdf = gpd.read_file(shapefile_path)

        # Get the bounds of the shapefile
        minx, miny, maxx, maxy = gdf.total_bounds
        center = [(miny + maxy) / 2, (minx + maxx) / 2]

        # Convert GeoDataFrame to GeoJSON
        geojson_data = gdf.to_json()

        print(f'Center: {center}, Bounds: {[minx, miny, maxx, maxy]}')  # Debugging output
        print(gdf.head())            # Check the first few rows of the data

        # Pass the center, bounds, and GeoJSON data to the frontend
        return render_template('index.html', center=center, bounds=[minx, miny, maxx, maxy], geojson_data=geojson_data)
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred. Please check the console for details."

if __name__ == '__main__':
    print("Hello")
    app.run(debug=True)
