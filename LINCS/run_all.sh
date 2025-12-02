#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <file>"
    echo "Provide the path to the configuration file"
    exit 1
fi

CONFIG="$1"

if [ ! -f "$CONFIG" ]; then
    echo "Error: $CONFIG does not exist."
    exit 1
fi

###------------------------------------SETUP--------------------------------------------------### 
mkdir moa-classification
cp -r resnet_model_helpers cp_resnet_train_pred.py moa-classification

export job_dir=$(pwd)
cd /scratch/appillai
mkdir $exp_name
cp -r cell-painting-devbench ./$exp_name
cd ./$exp_name/cell-painting-devbench/LINCS
mv $job_dir/*.py $job_dir/moa-classification ./
cp /scratch/appillai/features/$feat_name ./

###-------------------------------------------------------------------------------------------###


echo "STEP 2 - Feature aggregation"
python 02-lincs-well-aggregation-sphering-vits.py --config $CONFIG

echo "STEP 3 - Feature and metadata alignment"
python 03-align-cellprofiler-profiles.py --config $CONFIG

echo "STEP 4 - Create training and test partitions"
python 04-train-test-split.py --config $CONFIG

echo "STEP 5 - Train models"
python 05-moa-classification.py --config $CONFIG

echo "STEP 6 - Evaluate performance"
python 06-moa-predictions-visualization.py --config $CONFIG

#saving results
mkdir ${exp_name}_output
mv celldino_ps8_ViTs ./${exp_name}_output 
zip -r $exp_name ${exp_name}_output
cp $exp_name.zip /scratch/appillai/feat_eval_outputs
rm -rf /scratch/appillai/$exp_name
