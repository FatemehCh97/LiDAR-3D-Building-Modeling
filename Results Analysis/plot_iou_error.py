import os
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

# Define file path (modify as needed)
BASE_DIR = "data/footprints"
RESULTS_DIR = "results"

osm_shapefile_path = os.path.join(BASE_DIR, "OSM_Z.shp")
footprint_attention_path = os.path.join(BASE_DIR, "Attention_Z_Error.shp")
footprint_unet3p_path = os.path.join(BASE_DIR, "Unet3p_Z_Error.shp")

# List of metrics and corresponding titles
metrics = ['median_err', 'max_error_', 'mode_error', 'range_erro', 'perc_error']
metric_titles = ['Median', 'Maximum', 'Mode', 'Range', '90th Percentile']

def calculate_iou(geometry1, geometry2):
    """Calculate Intersection over Union (IoU) between two geometries."""
    intersection = geometry1.intersection(geometry2).area
    union = geometry1.union(geometry2).area
    return intersection / union if union != 0 else 0

def plot_iou_error_mean(footprint_attention_path, footprint_unet3p_path, metric):
    """Plots mean error vs. IoU for different models."""
    
    # Read shapefiles
    osm_footprints = gpd.read_file(osm_shapefile_path)
    attention_footprints = gpd.read_file(footprint_attention_path)
    unet3p_footprints = gpd.read_file(footprint_unet3p_path)

    # Convert 'osm_id' column to int64
    osm_footprints['osm_id'] = osm_footprints['osm_id'].astype('int64')

    # Merge data based on 'osm_id'
    merged_attention = osm_footprints.merge(attention_footprints, left_on='osm_id',
                                            right_on='OSM_ID', suffixes=('_osm', '_model'))
    merged_unet3p = osm_footprints.merge(unet3p_footprints, left_on='osm_id',
                                         right_on='OSM_ID', suffixes=('_osm', '_model'))

    # Compute IoU values
    merged_attention['iou'] = merged_attention.apply(lambda row: calculate_iou(row['geometry_osm'], row['geometry_model']), axis=1)
    merged_unet3p['iou'] = merged_unet3p.apply(lambda row: calculate_iou(row['geometry_osm'], row['geometry_model']), axis=1)

    # Define IoU bins and compute mean errors
    iou_bins = np.arange(0, 1.1, 0.1)
    mean_errors_attention = []
    mean_errors_unet3p = []

    for lower_bound, upper_bound in zip(iou_bins[:-1], iou_bins[1:]):
        bin_data_attention = merged_attention[(merged_attention['iou'] >= lower_bound) & (merged_attention['iou'] < upper_bound)]
        bin_data_unet3p = merged_unet3p[(merged_unet3p['iou'] >= lower_bound) & (merged_unet3p['iou'] < upper_bound)]
        mean_errors_attention.append(bin_data_attention[metric].mean())
        mean_errors_unet3p.append(bin_data_unet3p[metric].mean())

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(iou_bins[:-1], mean_errors_attention, color='blue', linestyle='-', marker='o', label='Attention U-Net')
    plt.plot(iou_bins[:-1], mean_errors_unet3p, color='red', linestyle='-', marker='o', label='UNet3p')
    plt.xlabel('Intersection over Union (IoU)')
    plt.ylabel(f'Mean {metric_titles[metrics.index(metric)]} Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title(f'Mean {metric_titles[metrics.index(metric)]} Error vs. IoU')
    plt.ylim(0, 6)

    # Save the figure
    output_path = os.path.join(RESULTS_DIR, f"iou_error_mean_{metric}.jpg")
    os.makedirs(RESULTS_DIR, exist_ok=True)  # Ensure directory exists
    plt.savefig(output_path, dpi=300)
    plt.show()

# Run the script for all metrics
for metric in metrics:
    plot_iou_error_mean(footprint_attention_path, footprint_unet3p_path, metric)
