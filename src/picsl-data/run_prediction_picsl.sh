#!/bin/bash
root="/home/mtduong/7T_invivo_project/data/nnUNet_prediction/3d_fullres"
operation_type=$1
cuda=$2
task_list="Task506_7Tgm-mergedlabels"

for task in $task_list; do
    prediction_task_folder="${root}/${task}/picsl-data"
    python3 ~/7T_invivo_project/src/prediction/data-prep-picsl-data.py -t ${task} -o ${operation_type}
    CUDA_VISIBLE_DEVICES=$cuda nnUNet_predict -i ${prediction_task_folder}/${operation_type}/input -o ${prediction_task_folder}/${operation_type}/output -t ${task} -m 3d_fullres --disable_mixed_precision -f all
    python3 ~/gmailsendmail/main.py -S "${task} Prediction on ${operation_type} done"
done
