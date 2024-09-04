# cell-painting-devbench
## Overview
This repository provides a step-by-step guide to successfully run the Dino4Cells_analysis with a focus on the dino-cp-analysis part. This project aims to use self-supervised learning methods (like DINO) to analyze biological imaging data and compare its performance with traditional feature extraction methods.

## Installation Guide
### 1. Clone the Repository
First, clone the repository to your local machine using Git:
```
git clone https://github.com/CaicedoLab/cell-painting-devbench.git
cd cell-painting-devbench

```

### 2. Set Up the Environment
Ensure you have `conda` installed. Create a new conda environment with the required dependencies:
```
conda create --name newenv python=3.11.7
conda activate newenv

```

### 3. Install Required Packages
Install the necessary Python packages using `conda`:
```
conda install tensorflow numpy pandas matplotlib
```
If you encounter version conflicts, update conda:
```
conda update -n base -c defaults conda  
```
If you still encounter issues with missing modules, install the packages using `pip`:
``` 
pip install scikit-learn pandas numpy iterative-stratification
```

### 4. Install TensorFlow and GPU Setup
Verify your GPU setup (if applicable):
```
nvidia-smi
```
Upgrade `pip` and reinstall TensorFlow based on your setup:
```
pip install --upgrade pip

# For GPU users
pip install tensorflow[and-cuda]

# For CPU users(prefer this)
pip install tensorflow
```
If you encounter errors related to `tensorflow.keras`, reinstall TensorFlow with a specific version:
```
pip uninstall keras
pip uninstall tensorflow
pip install tensorflow==2.12.0
```

### 5. Download and Prepare Data
Ensure that the following data files are available in the data directory:
`_cellprofiler_final.csv`
`_CNN_final.csv`
`_dino_final.csv`
`split_moas_cpds_final.csv`

### 6. Run Jupyter Notebooks
Launch Jupyter Notebook to start running the provided notebooks:
```
jupyter notebook
```
### 7. Execute the Notebooks
Open and run the following notebooks in order:

`06-train-test-split.ipynb`: Splits the dataset into training and testing sets based on compounds.
`07-moa-classification`: Trains models and predicts the mechanism of action (MOA) using different feature sets.
`08-moa-predictions-visualization.ipynb`: Visualizes the predictions made by the MOA classification models.# check
