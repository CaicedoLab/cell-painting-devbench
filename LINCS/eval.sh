#!/bin/bash

#set experiment name (add .zip at the end). BE SURE to change transfer_output_files (train.sub) value to exp_name
exp_name='test1.zip'

mv 01-feature-extraction.py 02-lincs-well-aggregation-sphering-vits.py 03-align-cellprofiler-profiles.py 04-train-test-split.py 06-moa-predictions-visualization.py /scratch/appillai/cell-painting-devbench1/LINCS
cd /scratch/appillai/cell-painting-devbench1/LINCS

#python 01-feature-extraction.py NOT CONFIGURED YET

python 02-lincs-well-aggregation-sphering-vits.py

python 03-align-cellprofiler-profiles.py

python 04-train-test-split.py

cd 05-moa-classification
./run_models.sh
cd ..

python 06-moa-predictions-visualization.py

rm cp_level4_cpd_replicates.csv.gz
mkdir output
mv celldino_ps8_ViTs ./output
mkdir celldino_ps8_ViTs
cp ./data/cp_CNN_final.csv celldino_ps8_ViTs

cd 05-moa-classification
mv model ../output

cd ..
zip -r $exp_name output
rm -rf output
