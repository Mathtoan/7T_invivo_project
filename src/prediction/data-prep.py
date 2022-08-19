import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import argparse
from random import shuffle

#%% Parser
parser = argparse.ArgumentParser(description='Data preparation for nnUNet training.')
parser.add_argument('-d', '--raw_data_folder', type=str, required=True,
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

#%% Read and write nifti functions
def read_nifti(filepath_image):

    img = nib.load(filepath_image)
    image_data = img.get_fdata()

    return image_data, img

def save_nifti(image, filepath_name, img_obj):

    img = nib.Nifti1Image(image, img_obj.affine, header=img_obj.header)
    nib.save(img, filepath_name)

#%% Setting up path
#TODO : custom mkdir so we can reset the folder
input_folder = join(prediction_data_root, model, task_name, 'input')
output_folder = join(prediction_data_root, model, task_name, 'output')
maybe_mkdir_p(input_folder)
maybe_mkdir_p(output_folder)

#%% Putting data to input folder
predicted_cases = subdirs(raw_data_folder, join=False)
num_subjects = len(predicted_cases) 


for i in range(len(predicted_cases)):
    subject = predicted_cases[i]
    ID, unique_name = subject, subject #TODO
    
    im_file_name = unique_name + "_T1w_7T_Preproc.nii.gz"
    raw_data_file = join(raw_data_folder, unique_name, im_file_name)

    input_image_file = join(input_folder, ID) # do not specify a file ending! This will be done for you
    input_image_file = input_image_file + "_%04.0d.nii.gz" % 0 # for now, end of file is 0000 because there is only one modality

    image_data, img_obj = read_nifti(raw_data_file)
    save_nifti(image_data, input_image_file, img_obj)
