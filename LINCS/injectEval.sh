#!/bin/bash

source 02env.sh

#injecting env variable values into eval.sub
sed -i -e "s/exp_name_inject/${exp_name}/g" -e "s/feat_name_inject/${feat_name}/g" -e "s/model_name_inject/${model_name}/g" 'eval.sub'
