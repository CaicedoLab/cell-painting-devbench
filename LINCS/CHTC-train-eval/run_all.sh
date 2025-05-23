#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <file>"
    echo "Provide the path to the configuration file"
    exit 1
fi

CONFIG="$1"
echo $CONFIG

if [ ! -f "$CONFIG" ]; then
    echo "Error: $file does not exist."
    exit 1
fi


export job_dir=$(pwd)

cd /scratch/appillai
mkdir $exp_name
cp -r cell-painting-devbench ./$exp_name
cp -r data ./$exp_name/cell-painting-devbench/LINCS
cd ./$exp_name/cell-painting-devbench/LINCS
mv $job_dir/*.py $job_dir/moa-classification $job_dir/config.json ./
mv $job_dir/utils ../
cp /scratch/appillai/features/$feat_name ./


echo "STEP 2 - Feature aggregation"
python 02-lincs-well-aggregation-sphering-vits.py --config $CONFIG

echo "STEP 3 - Feature and metadata alignment"
python 03-align-cellprofiler-profiles.py --config $CONFIG

echo "STEP 4 - Create training and test partitions"
python 04-train-test-split.py --config $CONFIG

cp -r celldino_ps8_ViTs/model_data/* moa-classification/celldino_ps8_ViTs/model_data

echo "STEP 5 - Train models"
python 05-moa-classification.py --config $CONFIG

cp -r moa-classification/celldino_ps8_ViTs/predictions ./celldino_ps8_ViTs

echo "STEP 6 - Evaluate performance"
python 06-moa-predictions-visualization.py --config $CONFIG

mkdir ${exp_name}_output
mv celldino_ps8_ViTs ./${exp_name}_output 
zip -r $exp_name ${exp_name}_output
cp $exp_name.zip /scratch/appillai/feat_eval_outputs

rm -rf /scratch/appillai/$exp_name
