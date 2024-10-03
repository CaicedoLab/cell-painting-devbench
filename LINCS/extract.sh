source env.txt
cd /scratch/appillai/DINOv2_CellPainting/dinov2 #need to create dir

PYTHONPATH=. torchrun --nproc_per_node=8 dinov2/eval/extract_features.py --config-file dinov2/configs/train/vits8_cellpainting.yaml --dataset CombinedDataset:root=/scratch/appillai/datasets/LINCS-DINO/max_concentration_set/:metadata_path=sc-metadata.csv --pretrained-weights /scratch/appillai/models/uw_dinov2_vits_teacher_checkpoint.pth --output-dir /scratch/appillai/features

echo "export exp_name=${exp_name}" > 02env.sh
echo "export feat_name=${feat_name}" >> 02env.sh
echo "export model_name=${model_name}" >> 02env.sh
