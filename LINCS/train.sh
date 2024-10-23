source env.txt

mv vits8_cellpainting.yaml /scratch/appillai/DINOv2_CellPainting/dinov2

cd /scratch/appillai
mkdir ${exp_name}_training 
cp -r DINOv2_CellPainting ${exp_name}_training
cd ${exp_name}_training/DINOv2_CellPainting/dinov2 

mkdir test

#set env variables
export PYTHONPATH=. 
export WANDB_API_KEY=61bf6d44e974b73170041842d514538f0ab4ed00 

#run training. Set gpus, provide config file, path to dataset
torchrun --nproc_per_node=1 ./dinov2/train/train.py --config-file ./vits8_cellpainting.yaml --output-dir ./test/ --experiment test train.dataset_path=CombinedDataset:root=/scratch/appillai/datasets/max_concentration_set/:metadata_path=/scratch/appillai/datasets/max_concentration_set/cs_sample1k.csv

cd ./test
mv *final* $weights_name
mv $weights_name /scratch/appillai/models

cd /scratch/appillai
rm -rf ${exp_name}_training
