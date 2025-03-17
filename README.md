# LiDAR-3D-Building-Modeling
This repository contains the Python codes and workflows used in the paper:
    "LOD1 3D City Model from LiDAR: The Impact of Segmentation Accuracy on Quality of Urban 3D Modeling and Morphology Extraction."

🔹 **Overview of the Work** \
We detect building footprints from LiDAR data using deep learning models and generate LOD1 3D models using various height estimation methods (e.g., median, 90th percentile). We also evaluate how footprint accuracy affects 3D model quality and the extraction of morphological parameters such as building area and exterior wall surface area.


📂 **Repository Structure**

`Building_Footprint_Detection/`

- Contains Python notebooks for extracting building footprints from LiDAR data using deep learning models (U-Net, Attention U-Net, U-Net3+, DeepLabV3+). Includes model implementation, training, evaluation, and performance analysis.
- Also features transfer learning techniques to adapt the models to the Netherlands dataset.

`FME_Workflows/`

- Contains FME-based workflows for generating LOD1 3D building models using LiDAR-derived footprints.

`Results_Analysis/`

- Python scripts used for generating statistical charts and performance metrics for result analysis. Used to assess segmentation accuracy, 3D model quality, and morphology extraction performance.
