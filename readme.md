# 7T invivo project
# Table of Contents
- [Requirement](#requirement)
- [Usage](#usage)
    - [Training](#training)
        - [Data preparation](#data-preparation)
        - [Running training](#running-training)
    - [Prediction](#prediction)

# Requirement
[nnUNet](https://github.com/MIC-DKFZ/nnUNet#run-inference) *(look at installation documention)*

# Usage
## **Training**
### ***Data preparation***
This part is to fullfill the requirement of nnUNet.

```bash
python3 src/data-prep.py -t TaskXXX_MYTASK -d DATASET_PATH -l LABEL_JSON_FILE_PATH
```
Works with this folder structure for dataset.

    my_dataset
    ├── 20200908x1133
    │   ├── 20200908x1133_3TSegTo7TDeformed.nii.gz
    │   ├── 20200908x1133_3Tto7THead0GenericAffine.mat
    │   ├── 20200908x1133_3Tto7THeadWarped.nii.gz
    │   ├── 20200908x1133_InitialAffineWarped.nii.gz
    │   ├── 20200908x1133_T1w_3T_PreProc.nii.gz
    │   ├── 20200908x1133_T1w_3T_PreProc_BrainMask.nii.gz
    │   ├── 20200908x1133_T1w_7T_Preproc.nii.gz
    │   ├── 20200908x1133_T1w_7T_Preproc_BrainMask.nii.gz
    │   └── 20200908x1133_T1w_7T_Preproc_BrainMaskedApplied.nii.gz
    ├── 20201022x1146
    │   ├── 20201022x1146_3TSegTo7TDeformed.nii.gz
    │   ├── 20201022x1146_3Tto7THead0GenericAffine.mat
    │   ├── 20201022x1146_3Tto7THeadWarped.nii.gz
    │   ├── 20201022x1146_T1w_3T_PreProc.nii.gz
    │   ├── 20201022x1146_T1w_3T_PreProc_BrainMask.nii.gz
    │   ├── 20201022x1146_T1w_7T_Preproc.nii.gz
    │   └── 20201022x1146_T1w_7T_Preproc_BrainMask.nii.gz
    ├── 20201201x1636
    │   ├── 20201201x1636_3TSegTo7TDeformed.nii.gz
    │   ├── 20201201x1636_3Tto7THead0GenericAffine.mat
    │   ├── 20201201x1636_3Tto7THeadWarped.nii.gz
    │   ├── 20201201x1636_T1w_3T_PreProc.nii.gz
    │   ├── 20201201x1636_T1w_3T_PreProc_BrainMask.nii.gz
    │   ├── 20201201x1636_T1w_7T_Preproc.nii.gz
    │   └── 20201201x1636_T1w_7T_Preproc_BrainMask.nii.gz
    ├── ...
Note that we only need `[ID]_3TSegTo7TDeformed.nii.gz`, `[ID]_T1w_7T_Preproc.nii.gz` (and `[ID]_3TSegTo7TDeformed.nii.gz` if we want to use mask with the `-m` mask)

(type `python3 src/data-prep.py --help` for more information)

If label management (merging labels or removing labels):

```bash
python3 src/training/data-prep-label.py -t TaskXXX_MYTASK -r LABELS_TO_REMOVE -R REFERENCE_LABEL_FOR_MERGING -M LABELS_TO_MERGE
```
### ***Running training***
1) Preprocessing
```bash
nnUNet_plan_and_preprocess -t TASK_NUMBER --verify_dataset_integrity
```
2) Training
```bash
CUDA_VISIBLE_DEVICES=X nnUNet_train 3d_fullres nnUNetTrainerV2 MY_TASK 'all' --npz
```


## **Prediction**
***Data preparation***
- running prediction
```bash
nnUNet_predict -i INPUT_FOLDER -o /OUTPUT_FOLDER -t TASK_NUMBER -m 3d_fullres --disable_mixed_precision -f all
```