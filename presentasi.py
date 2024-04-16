# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 19:08:28 2024

@author: vico
"""

# %% importing libraries

# data preparation
import rasterio
import os
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely import Polygon
from rasterio.mask import mask
import matplotlib.pyplot as plt

# feature engineering
import seaborn as sns
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from scipy import stats

# training
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from itertools import combinations

#evaluation
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

script_directory = os.path.dirname(os.path.abspath(__file__))

# %% importing raster data

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

# %% labelling untuk membuat dataset siap diolah

geojson_path = script_directory + "/geojson/"

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
    
    print("memproses file " + filename)
    
    #tipenya multipoints, kita baca pakai geopandas
    multipoints_gdf = gpd.read_file(geojson_path + filename)
    
    #crs untuk tiap band sama
    multipoints_gdf = multipoints_gdf.to_crs(b1_src_20.crs)
    
    for index, kategori in multipoints_gdf.iterrows():
        multipoint_geometry = kategori['geometry'] # ini berisi koodinat-koordinat long lat
        
        if index == 0:
            label = "bangunan"
        elif index == 1:
            label = "area_hijau"
        else:
            label = "air"
            
        for point in multipoint_geometry.geoms:
            #mengambil lon lat tiap koordinatnya
            lon, lat = point.x, point.y
            #konversi lon lat ke raster
            x_raster, y_raster = b1_src_20.index(lon, lat)
            
            try:
                #menambahkan semua band dengan value yang sesuai dari raster asli, kemudian menambahkan label yang sesuai
                B1_output.append(B1_20[x_raster][y_raster])
                B2_output.append(B2_20[x_raster][y_raster])
                B3_output.append(B3_20[x_raster][y_raster])
                B4_output.append(B4_20[x_raster][y_raster])
                B5_output.append(B5_20[x_raster][y_raster])
                B6_output.append(B6_20[x_raster][y_raster])
                B7_output.append(B7_20[x_raster][y_raster])
                B8A_output.append(B8A_20[x_raster][y_raster])
                B11_output.append(B11_20[x_raster][y_raster])
                B12_output.append(B12_20[x_raster][y_raster])
                labels_output.append(label)
            except:
                #menghitung jumlah label yang keluar raster
                out_of_bound_count += 1
                                
    print("koordinat2 yang out of bound :" + str(out_of_bound_count))

#mengoutputnya jadi dataset yang siap diolah
print("\nmemproses pembuatan file excel..")

output_filename = 'dataset_satelit_presentasi'
out_df = pd.DataFrame({'B1': B1_output, 'B2': B2_output, 'B3': B3_output, 'B4': B4_output, 'B5': B5_output, 'B6': B6_output, 'B7': B7_output, 'B8': B8A_output, 'B11': B11_output, 'B12': B12_output, 'jenis_lahan': labels_output})
out_df = out_df.drop_duplicates()

out_df.to_excel(script_directory + '/output_labelling/' + output_filename + ".xlsx", index=False)

print("file excel dataset sudah berhasil dibuat..")

#%% feature engineering

df = pd.read_excel(script_directory + "/output_labelling/dataset_satelit_presentasi.xlsx")

#visualisasi anova
features = df.iloc[:, :-1]  # semua kolom kecuali kolom target
target = df.iloc[:, -1] 

num_categories = len(target.unique())

# get_colors = lambda n: ["#%06x" % random.randint(0, 0xFFFFFF) for _ in range(n)]
# my_palette_1 = get_colors(num_categories) 

my_palette_1 = ["blue","orange","green"]

for feature in features.columns:
    with plt.rc_context(rc = {'figure.dpi': 150, 'axes.labelsize': 9, 
                              'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                              'legend.fontsize': 8.5, 'legend.title_fontsize': 9}):
    
        fig_2, ax_2 = plt.subplots(2, 2, figsize = (15, 10)) 
        
        sns.kdeplot(ax = ax_2[0, 0], x = df[feature], linewidth = 1.5,
                    hue = df[target.name], common_norm = True,
                    fill = True, alpha = 0.4, palette = my_palette_1)
        
        sns.stripplot(ax = ax_2[0, 1], x = df[target.name], s = 2,
                      y = df[feature], hue=df[target.name], legend=False, alpha = 1)
        
        ax_2[1, 0].set_visible(False)
        ax_2[1, 1].set_visible(False)
        
        plt.tight_layout(pad = 1.5)
        plt.show()

#%% hitung anova

def anova_test(features, target):
    # hitung F-value, p-value untuk setiap fitur
    f_values, p_values = f_classif(features, target)
    
    # buat dataframe untuk menyimpan hasil perhitungan
    anova_results = pd.DataFrame({'feature': features.columns, 'f_value': f_values, 'p_value': p_values})
    
    # sort berdasarkan p-value
    anova_results = anova_results.sort_values(by='f_value', ascending=False)
    
    # print 5 fitur dengan p-value terendah
    print(anova_results.head())
    print()
    
    alpha = 0.05
    
    # Mencetak hasil perbandingan antara nilai F dan nilai kritis F
    for i, row in anova_results.iterrows():
        f_crit = stats.f.ppf(1 - alpha, len(features.columns) - 1, len(features) - len(features.columns))
        if row['f_value'] > f_crit:
            print(f"Feature '{row['feature']}': F-value ({row['f_value']:.4f}) > F-critical ({f_crit:.4f}), significant.")
        else:
            print(f"Feature '{row['feature']}': F-value ({row['f_value']:.4f}) <= F-critical ({f_crit:.4f}), not significant.")
    
    print() 
   
    # Membuat larik boolean yang menunjukkan apakah nilai F lebih besar dari nilai kritis F
    significant_f = anova_results['f_value'] > f_crit

    # Memilih fitur yang memenuhi kriteria F-value > F-critical
    selected_features = features.columns[significant_f]

    
    print("Selected features:", selected_features)
    print()
    
    return selected_features

selected_features = anova_test(features, target)

#%% pilih fitur-fitur yang korelasinya terbaik dengan label
def select_k_best(features, target, k):
    # Membuat objek SelectKBest
    selector = SelectKBest(score_func=f_classif, k=k)
    
    # Fit dan transformasi fitur
    features_selected = selector.fit_transform(features, target)
    
    # Mendapatkan indeks fitur terpilih
    selected_indices = selector.get_support(indices=True)
    
    # Mendapatkan nama fitur terpilih
    selected_features = features.columns[selected_indices]
    
    return selected_features

# Jumlah fitur yang ingin dipilih
k = 3
selected_features_k_best = select_k_best(features, target, k)
print(f"Selected {k} best features based on SelectKBest: {selected_features_k_best}")

#%% cek pearson correlation antar band

df_cor = features.corr()
plt.figure(figsize=(10,10))
sns.heatmap(df_cor, annot=True)
plt.show()

feature_terpilih = ["B1", "B2", "B6", "B11"]

#%% training

train_df = pd.read_excel(script_directory + "/output_labelling/dataset_satelit_presentasi.xlsx")

def train(x_train, y_train):
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(x_train, y_train)
    return rf_classifier

def predict_real_data(rfc, data, x_coor, y_coor):
    y_pred = rfc.predict(data)
    results_df = pd.DataFrame({'x': x_coor, 'y': y_coor, 'jenis_lahan': y_pred})
    nama_file = "prediksi_real_presentasi"
    results_df.to_excel(script_directory + '/prediction_result/presentasi/' + nama_file + '.xlsx', index=False)
    
    return y_pred

#%% lakukan training
feature_terpilih = ["B1", "B2", "B6", "B11"]

x = train_df[feature_terpilih]
y = train_df['jenis_lahan']
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

random_forest_model = train(x_train, y_train)

#%% lakukan prediksi untuk evaluasi model

y_prediction = random_forest_model.predict(x_test)
y_actual = y_test

results_df = pd.DataFrame({'Actual': y_actual, 'Predicted': y_prediction})

results_df.to_excel(script_directory + '/prediction_result/presentasi/' + "prediksi_presentasi.xlsx", index=False)

#%% lakukan evaluasi

prediction_df = pd.read_excel(script_directory + "/prediction_result/presentasi/prediksi_presentasi.xlsx")

def evaluation_function(prediction_array, truth_array):

    # Compute confusion matrix
    cm = confusion_matrix(truth_array, prediction_array)
    classes = set(truth_array.unique())
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Compute metrics
    accuracy = accuracy_score(truth_array, prediction_array)
    precision = precision_score(truth_array, prediction_array, average=None)
    recall = recall_score(truth_array, prediction_array, average=None)
    f1 = f1_score(truth_array, prediction_array, average=None)
    
    # Print metrics
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("Precision")
    for i, cls in enumerate(classes):
        print("Class", cls, ":", "{:.2f}%".format(precision[i] * 100))
    print("Recall")
    for i, cls in enumerate(classes):  
        print("Class", cls, ":", "{:.2f}%".format(recall[i] * 100))
    print("F1 score")
    for i, cls in enumerate(classes):
        print("Class", cls, ":", "{:.2f}%".format(f1[i] * 100))

    f1_weighted = f1_score(truth_array, prediction_array, average='weighted')
    print("F1 Score (weighted): {:.2f}%".format(f1_weighted * 100))
    print()
    
    
prediction_array = prediction_df['Predicted']
actual_array = prediction_df['Actual']
evaluation_function(prediction_array, actual_array)


#%% coba prediksi yang asli

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

polygon = Polygon(test_geojson[0]["coordinates"][0])
polygon_gdf = gpd.GeoDataFrame(geometry=[polygon])
polygon_gdf.crs = "EPSG:4326"
polygon_gdf_reprojected = polygon_gdf.to_crs(b2_src_20.crs)

#%% clip rasternya ke koordinat di test_geojson
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

#%% pembuatan dataset clipping yang belum dilabel (menghiraukan 0)

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

#%% buat file excelnya 

output_filename = 'dataset_presentasi_real'
out_df = pd.DataFrame({'B1': output_arr[0], 'B2': output_arr[1], 'B3': output_arr[2], 'B4': output_arr[3], 'B5': output_arr[4], 'B6': output_arr[5], 'B7': output_arr[6], 'B8': output_arr[7], 'B11': output_arr[8], 'B12': output_arr[9], 'x': output_arr[10], 'y': output_arr[11]})
out_df.to_excel(script_directory + '/output_labelling/' + output_filename + ".xlsx", index=False)
print("file excel sudah dibuat")

#%% lakukan prediksi terhadap data asli

predict_data = pd.read_excel(script_directory + '/output_labelling/dataset_presentasi_real.xlsx')
hasil = predict_real_data(random_forest_model, predict_data[feature_terpilih], predict_data['x'], predict_data['y'])


#%% import datasetnya
hasil_prediksi = pd.read_excel(script_directory +"/prediction_result/presentasi/prediksi_real_presentasi.xlsx")


#%% print peta asli
normalized_b2 = clipped_b2[0] / clipped_b2[0].max() * 255
normalized_b3 = clipped_b3[0] / clipped_b3[0].max() * 255
normalized_b4 = clipped_b4[0] / clipped_b4[0].max() * 255


rgb_image = np.dstack((normalized_b4, normalized_b3, normalized_b2)).astype(np.uint8)
rgb_raw = np.dstack((clipped_b2[0], clipped_b3[0], clipped_b4[0]))

plt.figure(figsize=(20, 12))  # Set width to 10 inches, height to 6 inches
plt.imshow(rgb_image)


#%% print peta hasil klasifikasi

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
        
