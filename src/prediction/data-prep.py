import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import argparse
from random import shuffle

#%% Parser
parser = argparse.ArgumentParser(description='Data preparation for nnUNet training.')
parser.add_argument('-d', '-raw_data_folder', type=str, required=True,
                    help='Folder where the data to predict are.')
parser.add_argument('-r', '--prediction_data_root', type=str, default='/data/mtduong/7T_invivo_project/nnUNet_prediction')
# parser.add_argument("-i", '--input_folder', type=str,
#                     help="Must contain all modalities for each patient in the correct"
#                          " order (same as training). Files must be named "
#                          "CASENAME_XXXX.nii.gz where XXXX is the modality "
#                          "identifier (0000, 0001, etc)", required=True)
# parser.add_argument('-o', "--output_folder", type=str, required=True,
#                     help="folder for saving predictions")
parser.add_argument('-t', '--task_name', type=str, required=True,
                    help='task name or task ID, required.')
parser.add_argument('-m', '--model', default="3d_fullres",
                    help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres")

args = parser.parse_args()

#%% Parameter
raw_data_folder = args.raw_data_folder
prediction_data_root = args.prediction_data_root
# input_folder = args.input_folder
# output_folder = args.output_folder
task_name = args.task_name
model = args.model