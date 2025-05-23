source env.txt
cd /scratch/appillai/DINOv2_CellPainting/dinov2 

#PYTHONPATH=. torchrun --nproc_per_node=9 dinov2/eval/extract_features.py --config-file /scratch/appillai/DINOv2_CellPainting/dinov2/vits8_cellpainting.yaml --dataset CombinedDataset:root=/scratch/appillai/datasets/max_concentration_set/:metadata_path=sc-metadata.csv --pretrained-weights /scratch/appillai/models/$exp_name/model_final --output-dir /scratch/appillai/features  

echo "export exp_name=${exp_name}" > 02env.sh
echo "export feat_name=${feat_name}" >> 02env.sh
echo "export model_name=${model_name}" >> 02env.sh

