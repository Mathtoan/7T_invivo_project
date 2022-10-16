#!/bin/bash
root="/home/mtduong/7T_invivo_project/data/nnUNet_prediction/3d_fullres"
operation_type=$1
cuda=$2

# prediction_task_folder="${root}/Task501_7Tgm/picsl-data"
# python3 ~/7T_invivo_project/src/prediction/data-prep-picsl-data.py -t Task501_7Tgm -o ${operation_type}
# CUDA_VISIBLE_DEVICES=$cuda nnUNet_predict -i ${prediction_task_folder}/${operation_type}/input -o ${prediction_task_folder}/${operation_type}/output -t 504 -m 3d_fullres --disable_mixed_precision -f all
# python3 ~/gmailsendmail/main.py -S "Task 504 Prediction on ${operation_type} done"

prediction_task_folder="${root}/Task504_7Tgm-lesslabel/picsl-data"
python3 ~/7T_invivo_project/src/prediction/data-prep-picsl-data.py -t Task504_7Tgm-lesslabel -o ${operation_type}
CUDA_VISIBLE_DEVICES=$cuda nnUNet_predict -i ${prediction_task_folder}/${operation_type}/input -o ${prediction_task_folder}/${operation_type}/output -t 504 -m 3d_fullres --disable_mixed_precision -f all
python3 ~/gmailsendmail/main.py -S "Task 504 Prediction on ${operation_type} done"

prediction_task_folder="${root}/Task506_7Tgm-mergedlabels"
python3 ~/7T_invivo_project/src/prediction/data-prep-picsl-data.py -t Task506_7Tgm-mergedlabels -o ${operation_type}
CUDA_VISIBLE_DEVICES=$cuda nnUNet_predict -i ${prediction_task_folder}/${operation_type}/input -o ${prediction_task_folder}/${operation_type}/output -t 504 -m 3d_fullres --disable_mixed_precision -f all
python3 ~/gmailsendmail/main.py -S "Task 506 Prediction on ${operation_type} done"


