# cell-painting-devbench
## Overview
This repository provides a step-by-step guide to successfully run the Dino4Cells_analysis with a focus on the dino-cp-analysis part. This project aims to use self-supervised learning methods (like DINO) to analyze biological imaging data and compare its performance with traditional feature extraction methods.

## Installation Guide
### 1. Clone the Repository
clone the repository to your local machine using Git:
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
conda install numpy pandas matplotlib
conda update -n base -c defaults conda  
```
Install the following packages using `pip`:
``` 
pip install --upgrade pip
pip install scikit-learn pandas numpy iterative-stratification
pip install scikit-image tqdm seaborn adjustText
```

### 4. Install TensorFlow
The code in this benchmark requires TensorFlow 2.12. We recommed using the CPU version of the package to prevent issues with GPU drivers because TF 2.12 is an old version.
```
pip uninstall keras
pip uninstall tensorflow
pip install tensorflow==2.12.0
```

### 5. Execute the Notebooks
Open and run the following notebooks in LINCS dictionary.
