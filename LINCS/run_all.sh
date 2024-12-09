#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Usage: $0 <file>"
    echo "Provide the path to the configuration file"
    exit 1
fi

CONFIG="$1"

if [ ! -f "$CONFIG" ]; then
    echo "Error: $file does not exist."
    exit 1
fi

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
