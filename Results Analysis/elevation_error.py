import os
import geopandas as gpd
import numpy as np

# Define file path (modify as needed)
BASE_DIR = "data/footprints"
RESULTS_DIR = "results"

osm_shapefile_path = os.path.join(BASE_DIR, "OSM_Z.shp")
footprint_attention_path = os.path.join(BASE_DIR, "Attention_Z.shp")
footprint_unet3p_path = os.path.join(BASE_DIR, "Unet3p_Z.shp")

# List of elevation attributes to compute errors
elevation_metrics = ['z_median', 'z_max', 'z_mode', 'z_range', 'Q_90th_per']

def compute_errors(merged_gdf, metric):
    """Compute error and absolute error for the given elevation metric."""
    merged_gdf[f'{metric}_error'] = merged_gdf[f'{metric}_model'] - merged_gdf[f'{metric}_osm']
    merged_gdf[f'abs_{metric}_error'] = merged_gdf[f'{metric}_error'].abs()
    return merged_gdf

# Load data
osm_footprints = gpd.read_file(osm_shapefile_path)
attention_footprints = gpd.read_file(footprint_attention_path)
unet3p_footprints = gpd.read_file(footprint_unet3p_path)

# Convert 'osm_id' column to int64
osm_footprints['osm_id'] = osm_footprints['osm_id'].astype('int64')

# Merge data based on 'osm_id'
merged_attention = osm_footprints.merge(attention_footprints, left_on='osm_id', right_on='OSM_ID', suffixes=('_osm', '_model'))
merged_unet3p = osm_footprints.merge(unet3p_footprints, left_on='osm_id', right_on='OSM_ID', suffixes=('_osm', '_model'))

# Compute errors for each metric
for metric in elevation_metrics:
    merged_attention = compute_errors(merged_attention, metric)
    merged_unet3p = compute_errors(merged_unet3p, metric)

# Drop unnecessary error columns, keeping only absolute error columns
error_columns = ['z_error', 'z_median_error', 'z_max_error', 'z_mode_error', 'z_range_error']
merged_attention.drop(columns=error_columns, inplace=True, errors='ignore')
merged_unet3p.drop(columns=error_columns, inplace=True, errors='ignore')

# Rename absolute error columns
rename_dict = {
    'z_median_error_abs': 'median_error_abs',
    'z_max_error_abs': 'max_error_abs',
    'z_mode_error_abs': 'mode_error_abs',
    'z_range_error_abs': 'range_error_abs',
    'z_error_abs': 'perc_error_abs'
}

merged_attention.rename(columns=rename_dict, inplace=True)
merged_unet3p.rename(columns=rename_dict, inplace=True)

# Save processed shapefiles
os.makedirs(RESULTS_DIR, exist_ok=True)
attention_output_path = os.path.join(RESULTS_DIR, "Attention_Z_Error.shp")
unet3p_output_path = os.path.join(RESULTS_DIR, "Unet3p_Z_Error.shp")

merged_attention.to_file(attention_output_path)
merged_unet3p.to_file(unet3p_output_path)

print("Processing completed. Processed shapefiles saved in results directory.")
