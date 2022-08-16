
# export nnUNet_raw_data_base="/home/mtduong/7T_invivo_project/data/nnUNet_raw_data_base"
# export nnUNet_preprocessed="/home/mtduong/7T_invivo_project/data/nnUNet_preprocessed" # Should be on an SSD
# export RESULTS_FOLDER="/home/mtduong/7T_invivo_project/data/nnUNet_trained_models"

echo "DATA PREPARATION"
python3 /home/mtduong/7T_invivo_project/src/Task501/Task501-data_prep.py

echo "PLAN & PREPROCESS"
nnUNet_plan_and_preprocess -t 501 --verify_dataset_integrity

echo "TRAINING"
CUDA_VISIBLE_DEVICES=1 nnUNet_train 3d_fullres nnUNetTrainerV2 Task501_7Tgm 'all' --npz