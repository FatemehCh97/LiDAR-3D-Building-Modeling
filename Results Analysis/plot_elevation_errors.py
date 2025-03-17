import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# Define file path (modify as needed)
BASE_DIR = "data/footprints"
RESULTS_DIR = "results"

attention_path = os.path.join(BASE_DIR, "Attention_Z_Error.shp")
unet3p_path = os.path.join(BASE_DIR, "Unet3p_Z_Error.shp")


def load_shapefile(file_path):
    """Loads a shapefile into a GeoDataFrame."""
    if os.path.exists(file_path):
        return gpd.read_file(file_path)
    else:
        raise FileNotFoundError(f"File not found: {file_path}")


# Load the shapefiles
attention_footprints = load_shapefile(attention_path)
unet3p_footprints = load_shapefile(unet3p_path)

# List of metrics
metrics = ['median_err', 'max_error_', 'mode_error', 'range_erro', 'perc_error']

# Define titles for each metric
metric_titles = [
    'Median Elevation Error', 
    'Maximum Elevation Error', 
    'Mode Elevation Error', 
    'Range Elevation Error', 
    '90th Percentile Elevation Error'
]

# Define custom bin edges
bin_edges = np.array([0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, 2.50, 5.00, 10.00])


def plot_histogram(ax, data, metric, title, color, model_name):
    """Plots a histogram for a given metric."""
    counts, _, _ = ax.hist(data[metric], bins=bin_edges, color=color, alpha=0.7, label=model_name)
    ax.set_title(title)
    ax.set_xlabel('Error Value')
    ax.set_ylabel('Frequency')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(counts) + 50)
    ax.legend(loc='upper right')

    return counts


def print_histogram_stats(counts, metric_title, model_name):
    """Prints bin-wise distribution statistics for a histogram."""
    total = sum(counts)
    print(f"\n{metric_title} - {model_name}")
    for j in range(len(bin_edges) - 1):
        print(f"Bin Range: {bin_edges[j]:.2f} - {bin_edges[j+1]:.2f}, "
              f"Count: {counts[j]:.0f}, Percentage: {(counts[j] / total) * 100:.2f}%")


# Create figure with subplots
fig, axes = plt.subplots(2, len(metrics), figsize=(20, 12))

# Plot histograms for each metric
for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
    # Attention U-Net Model
    counts_attention = plot_histogram(axes[0, i], attention_footprints, metric, title, 'blue', 'Attention U-Net')
    print_histogram_stats(counts_attention, title, "Attention U-Net")

    # UNet3p Model
    counts_unet3p = plot_histogram(axes[1, i], unet3p_footprints, metric, title, 'red', 'UNet3p')
    print_histogram_stats(counts_unet3p, title, "UNet3p")

plt.tight_layout()

# Save the plot
output_path = os.path.join(RESULTS_DIR, "z_error_histogram.jpg")
plt.savefig(output_path, dpi=300)

plt.show()
