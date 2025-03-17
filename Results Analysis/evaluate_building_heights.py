import os
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Define file path (modify as needed)
BASE_DIR = "data/footprints"

osm_shapefile_path = os.path.join(BASE_DIR, "OSM_Z.shp")
unet3p_shapefile_path = os.path.join(BASE_DIR, "Unet3p_Z_Error.shp")
attention_shapefile_path = os.path.join(BASE_DIR, "Attention_Z_Error.shp")


def load_shapefile(file_path):
    """Loads a shapefile into a GeoDataFrame."""
    if os.path.exists(file_path):
        return gpd.read_file(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")


# Load the shapefiles
osm_gdf = load_shapefile(osm_shapefile_path)
unet3p_gdf = load_shapefile(unet3p_shapefile_path)
attention_gdf = load_shapefile(attention_shapefile_path)

# Convert 'osm_id' column to int64 in OSM dataset
osm_gdf['osm_id'] = osm_gdf['osm_id'].astype('int64')

# Select relevant columns from model shapefiles
columns_to_keep = ['OSM_ID', 'z_max', 'z_mean', 'z_median', 'z_mode', 'z_range', 'Q_90th_per']

# Merge OSM with U-Net3p data
merged_gdf = osm_gdf.merge(
    unet3p_gdf[columns_to_keep], 
    left_on='osm_id', right_on='OSM_ID', 
    suffixes=('_osm', '_unet3p')
).drop(columns='OSM_ID')

# Rename columns in attention_gdf before merging
attention_gdf = attention_gdf.rename(columns={col: col + '_attention' for col in columns_to_keep if col != 'OSM_ID'})

# Merge with Attention U-Net data
merged_gdf = merged_gdf.merge(
    attention_gdf, 
    left_on='osm_id', right_on='OSM_ID', 
    suffixes=('_unet3p', '_attention')
).drop(columns='OSM_ID')

# Rename percentile column for consistency
merged_gdf.rename(columns={
    'Q_90th_per_osm': 'z_percent_osm',
    'Q_90th_per_unet3p': 'z_percent_unet3p',
    'Q_90th_per_attention': 'z_percent_attention'
}, inplace=True)

# Function to compute and print metrics
def compute_metrics(true_values, pred_values, model_name, variable):
    """Computes and prints RMSE, MAE, and R² for a given model and variable."""
    rmse = np.sqrt(mean_squared_error(true_values, pred_values))
    mae = mean_absolute_error(true_values, pred_values)
    r2 = r2_score(true_values, pred_values)

    print(f"Variable: {variable}")
    print(f"Model: {model_name}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    print("------------------------")

# List of variables to evaluate
variables = ['z_median', 'z_max', 'z_mean', 'z_mode', 'z_range', 'z_percent']

# Compute metrics for each variable
for variable in variables:
    compute_metrics(merged_gdf[f'{variable}_osm'], merged_gdf[f'{variable}_unet3p'], 'U-Net3+', variable)
    compute_metrics(merged_gdf[f'{variable}_osm'], merged_gdf[f'{variable}_attention'], 'Attention U-Net', variable)
