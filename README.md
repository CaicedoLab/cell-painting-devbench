# Cell Painting Evaluation
## Overview
This repository provides a step-by-step guide to successfully run the LINCS Cell Painting evaluation using Cell-DINO. The LINCS Cell Painting evaluation code was originally developed by [Way et al. (2022)](https://www.cell.com/cell-systems/fulltext/S2405-4712(22)00402-1) in their morphology and transcriptional profiling study. The code in this evaluation is adapted from their implementation ([available in GitHub](https://github.com/broadinstitute/lincs-profiling-complementarity)) and focused only on the morphological profiling analysis.

### Training Cell-DINO models and feature extraction
This repository does not contain the code to train Cell-DINO models and does not contain code to extract the corresponding feature embeddings from images. Training and feature extraction need to be performed with the DINOv2 implementation provided with the HPA repository (link to the other repository to be added soon). Please, follow the instructions there to obtain a pre-trained Cell-DINO model for Cell Painting images, or to train your own Cell-DINO model.

The Cell Painting datasets used in this study are available in the [Cell Painting Gallery](https://github.com/broadinstitute/cellpainting-gallery/blob/main/README.md). To train the Cell-DINO model we employed the Combined Cell Painting dataset (accession number `cpg0019-moshkov-deepprofiler`) used in the [DeepProfiler study](https://www.nature.com/articles/s41467-024-45999-1). The dataset has 482GB of imaging data and about 8M single cells. Training with this dataset with 8 GPUs takes about 24 hours with a batch size of 256 and 1 million images per epoch for 100 epochs. For the evaluation part, images from the LINCS Cell Painting dataset were used (accession number `cpg0004-lincs`), which is a compound screen of approx. 1,500 FDA approved drugs. The full image collection is 61TB of imaging data, and we only used one dose at a time for analysis (out of six available doses). Feature extraction at the single-cell level with the Cell-DINO model on the LINCS dataset takes about 2 hours on 8 GPUs.

### Evaluation task
The main focus of this repository is to predict the mechanism of action of compounds and evaluate the ability of feature extraction methods to do this accurately. This repository assumes that features have been already computed, and provides the necessary code to aggregate, batch-correct, train classifiers, and evaluate performance. The whole procedure is a multi-step process which has been automated to be run in the command line without manual intervention. Running the evaluation from scratch can take about 2 hours using CPU compute only.

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

### 5. Execute the process
The process has 5 main steps, ennumerated from 2 to 6 (step 1 is feature extraction, which is not part of this repository). For each step, there is one script in the LINCS directory, An example JSON file with configuration parameters is provided, and the main parameter to adjust is the directory where the inputs and outputs are stored. As a baseline, this process uses precomputed CellProfiler features, which are automatically downloaded from the web by the scripts. 

For convinience, there is a shell script that runs all the steps one after the other. To run all steps use:
```
sh run_all.sh
```
If you need to run one or a few steps only, feel free to modify this script by commenting out the steps that you want to skip. Note that running each of the steps is as simple as calling Python with the corresponding script and the JSON configuration file as an input. This code is sufficient to reproduce the Cell Painting results reported in the Cell-DINO paper.
