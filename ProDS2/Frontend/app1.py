from flask import Flask, render_template, jsonify, send_file, request
import pandas as pd
# Geospatial processing packages
import geopandas as gpd

import re
import os
import sys
import urllib.parse
from PIL import Image
import zipfile
from io import BytesIO

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(script_dir)

import asyncio
import nest_asyncio



try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


app = Flask(__name__)

citarum_gdf = ""


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
        selected_details = selected_details.sort_values(by='kelas')

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


@app.route('/dta_image/<path:dta_name><image_type>')
def dta_image(dta_name, image_type):
    dta_name = urllib.parse.unquote(dta_name)

    new_path = os.path.join(os.path.dirname(script_dir), 'Images/')
    dta_name_cleaned = dta_name.replace(" ", "").replace("/", "_")

    if image_type == 'raw':
        image_path = os.path.join(new_path, f"{dta_name_cleaned}raw.png")
    elif image_type == 'classified':
        image_path = os.path.join(new_path, f"{dta_name_cleaned}classified.png")
    else:
        return jsonify({"error": "Invalid image type"}), 400

    # Cek apakah gambar ada
    if os.path.exists(image_path):
        # Buka gambar dan hapus background putih
        image = Image.open(image_path)
        image = image.convert("RGBA")
        data = image.getdata()

        new_data = []
        for item in data:
            # Ubah warna putih menjadi transparan
            if item[:3] == (255, 255, 255):
                new_data.append((255, 255, 255, 0))  # Transparan
            else:
                new_data.append(item)
        image.putdata(new_data)

        # Potong gambar untuk menghapus area transparan
        bbox = image.getbbox()  # Mendapatkan bounding box non-transparan
        if bbox:
            image = image.crop(bbox)  # Potong gambar ke area konten

        # Simpan gambar hasil edit ke disk
        edited_image_path = os.path.join(new_path, f"{dta_name_cleaned}_{image_type}_edited.png")
        image.save(edited_image_path, "PNG")

        return send_file(edited_image_path, mimetype='image/png')
    else:
        return jsonify({"error": "Image not found"}), 404

@app.route('/download_images/<path:dta_name>')
def download_images(dta_name):
    dta_name = urllib.parse.unquote(dta_name)
    new_path = os.path.join(os.path.dirname(script_dir), 'Images/')
    dta_name_cleaned = dta_name.replace(" ", "").replace("/", "_")

    # Lokasi file gambar
    raw_image_path = os.path.join(new_path, f"{dta_name_cleaned}_raw.png")
    classified_image_path = os.path.join(new_path, f"{dta_name_cleaned}_classified.png")

    print(raw_image_path)
    print(classified_image_path)
    print("aj")
    # Cek apakah file ada
    if not os.path.exists(raw_image_path) or not os.path.exists(classified_image_path):
        return jsonify({"error": "One or more images not found"}), 404

    # Buat file ZIP dalam memori
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zf:
        zf.write(raw_image_path, os.path.basename(raw_image_path))
        zf.write(classified_image_path, os.path.basename(classified_image_path))
    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name=f"{dta_name_cleaned}_images.zip"
    )

if __name__ == '__main__':
    print("Hello")
    nest_asyncio.apply()
    app.run(debug=True, threaded = False)
