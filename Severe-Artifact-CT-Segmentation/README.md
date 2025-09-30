Severe Artifact Conditions Fine Segmentation for Core CT Images Combining Radon Transform and DINOv2
📖 Project Description
This repository contains the official implementation of the paper "Severe Artifact Conditions Fine Segmentation for Core CT Images Combining Radon Transform and DINOv2" submitted to Computer & Geoscience.

Key Innovations:

Integration of Radon transform for artifact correction using directional characteristics

Utilization of DINOv2 vision foundation model for global context understanding

Adaptive learning framework for pore-mineral-matrix discrimination

Robust segmentation under severe artifact conditions while maintaining geological plausibility

🎯 Background
Accurate segmentation of core CT images is crucial for digital rock physics analysis. However, high-density minerals often cause severe artifacts that distort pore structures and reduce segmentation reliability. This work addresses these challenges by combining traditional image processing with modern foundation models.

📁 Project Structure
text

Core-CT-Segmentation/

├── data/                    # Data directory

├── pre_data/               # Data preprocessing (Radon transform)

├── visual_analysis/        # Result visualization and analysis

├── requirements.txt        # Python dependencies

└── README.md


🚀 Quick Start
Prerequisites
Python 3.8+
CUDA-capable GPU (recommended)

Installation

bash
pip install -r requirements.txt
Data Preparation
Datasets can be available upon reasonable request by contacting the corresponding author.

Place data in the data/ directory

Run preprocessing (Radon transform):

bash
cd pre_data
python radon_transform.py
Usage
Training and inference:


bash
bash mla_core_seg.sh
Note: Configure dataset paths in the scripts before execution.

Visualization:

bash
cd ../visual_analysis
python result_visualization.py

🔧 Key Features
Artifact Robustness: Effective handling of heavy-mineral-induced artifacts

Geological Consistency: Maintains pore structure continuity and geological plausibility

Adaptive Learning: Handles significant annotation variability

Comprehensive Pipeline: From preprocessing to visualization

📊 Results
Extensive experiments on synthetic and real core datasets demonstrate:

High segmentation accuracy under severe artifact conditions

Robust performance with significant annotation variations

Superior results compared to traditional methods

🙏 Acknowledgement
This work is based on Cross-Domain-Foundation-Model-Adaptation.
