import os
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

# Define file path (modify as needed)
BASE_DIR = "data/footprints"
RESULTS_DIR = "results"

osm_shapefile_path = os.path.join(BASE_DIR, "OSM_Z.shp")
footprint_unet3p_path = os.path.join(BASE_DIR, "Unet3p_Z_Error.shp")
footprint_attention_path = os.path.join(BASE_DIR, "Attention_Z_Error.shp")

def calculate_iou(geometry1, geometry2):
    """Calculate Intersection over Union (IoU) between two geometries."""
    intersection = geometry1.intersection(geometry2).area
    union = geometry1.union(geometry2).area
    return intersection / union if union != 0 else 0

def plot_iou_histogram(footprint_shapefile_path, title, ax, osm_footprints):
    """Plots IoU histogram for a given footprint shapefile."""
    model_footprints = gpd.read_file(footprint_shapefile_path)

    osm_footprints['osm_id'] = osm_footprints['osm_id'].astype('int64')

    merged_footprints = osm_footprints.merge(model_footprints, left_on='osm_id',
                                             right_on='OSM_ID', suffixes=('_osm', '_model'))

    # Compute IoU
    merged_footprints['iou'] = merged_footprints.apply(
        lambda row: calculate_iou(row['geometry_osm'], row['geometry_model']), axis=1
    )

    # Plot histogram
    hist, bins, _ = ax.hist(merged_footprints['iou'], bins=20, color='#1f77b4', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Intersection over Union (IoU)', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, linestyle='--', alpha=0.5)

    # Normalize histogram frequencies
    ax.bar(bins[:-1], hist / hist.max(), width=(bins[1] - bins[0]))

    return hist.max()

# Read the OSM shapefile
osm_footprints = gpd.read_file(osm_shapefile_path)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Plot IoU histograms
max_freq_unet3p = plot_iou_histogram(footprint_unet3p_path, 'U-Net3+ Model', ax1, osm_footprints)
max_freq_attention = plot_iou_histogram(footprint_attention_path, 'Attention U-Net Model', ax2, osm_footprints)

# Adjust y-axis limits
ax1.set_ylim(0, max_freq_unet3p)
ax2.set_ylim(0, max(max_freq_unet3p, max_freq_attention))

# Adjust tick label sizes
ax1.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

plt.tight_layout()

# Save the plot
output_path = os.path.join(RESULTS_DIR, "iou_histogram.jpg")
os.makedirs(RESULTS_DIR, exist_ok=True)
plt.savefig(output_path, dpi=300)

plt.show()
