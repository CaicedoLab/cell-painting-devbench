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

for k in {1..9}; do
    echo "STEP 5 - Train models"
    python 05-moa-classification.py --config $CONFIG --repeat

    echo "STEP 6 - Evaluate performance"
    python 06-moa-predictions-visualization.py --config $CONFIG
done
