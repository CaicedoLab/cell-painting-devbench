#!/bin/bash

source 02env.sh
cp eval.sub evalInject.sub

#injecting env variable values into eval.sub
sed -i -e "s/exp_name_inject/${exp_name}/g" -e "s/feat_name_inject/${feat_name}/g" -e "s/model_name_inject/${model_name}/g" 'evalInject.sub'

jq '.feature_path = "/scratch/appillai/features/'${feat_name}'"' config.json > temp.json && mv temp.json config.json