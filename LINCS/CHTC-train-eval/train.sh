source env.txt

mv vits8_cellpainting.yaml /scratch/appillai/DINOv2_CellPainting/dinov2

cd /scratch/appillai
mkdir ${exp_name}_training 
cp -r DINOv2_CellPainting ${exp_name}_training
cd ${exp_name}_training/DINOv2_CellPainting/dinov2 

mkdir /scratch/appillai/test

#set env variables
export PYTHONPATH=. 
export WANDB_API_KEY=61bf6d44e974b73170041842d514538f0ab4ed00 

#run training. Set gpus, provide config file, path to dataset
torchrun --nproc_per_node=1 ./dinov2/train/train.py --config-file ./vits8_cellpainting.yaml --output-dir ./scratch/appillai/test --experiment test train.dataset_path=CombinedDataset:root=/scratch/appillai/datasets/combined_set/:metadata_path=/scratch/appillai/datasets/combined_set/training_meta.csv

mkdir /scratch/appillai/models/$exp_name
cd /scratch/appillai/test
mv model_final* /scratch/appillai/models/$exp_name

cd /scratch/appillai
rm -rf ${exp_name}_training ./test
