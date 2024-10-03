#!/bin/bash

export job_dir=$(pwd)
cd /scratch/appillai
mkdir $exp_name
cp -r cell-painting-devbench1 data ./$exp_name
cd ./$exp_name/cell-painting-devbench1/LINCS
mv $job_dir/*.py ./
cp /staging/groups/caicedo_group/features_eval/$feat_name ./

python 02-lincs-well-aggregation-sphering-vits.py

python 03-align-cellprofiler-profiles.py

python 04-train-test-split.py

cd 05-moa-classification
./run_models.sh
cd ..

python 06-moa-predictions-visualization.py

mkdir ${exp_name}_output
mv celldino_ps8_ViTs ./${exp_name}_output

cd 05-moa-classification
mv model ../${exp_name}_output
cd ..

zip -r $exp_name ${exp_name}_output
cp $exp_name.zip /scratch/appillai/feat_eval_outputs
rm -rf /scratch/appillai/$exp_name
