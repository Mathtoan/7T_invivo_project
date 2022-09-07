import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import argparse
from random import shuffle
from glob import glob

#%% Parser
parser = argparse.ArgumentParser(description='Data preparation for nnUNet training.')
parser.add_argument('-d', '--dataset_folder', type=str, required=True,
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
parser.add_argument('-M', '--applymask', action='store_true',
                    help='Apply mask to image')
parser.add_argument('-m', '--model', default="3d_fullres",
                    help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres")
parser.add_argument('-p', '--preprocessed', action='store_true',
                    help="Choose if data is preprocessed or not")
parser.add_argument('-tr', '--train_set', default="/data/mtduong/7T_invivo_project/dataset/trainset1.json",
                    help="path of the train set json file.")

args = parser.parse_args()

#%% Parameter
dataset_folder = args.dataset_folder
prediction_data_root = args.prediction_data_root
# input_folder = args.input_folder
# output_folder = args.output_folder
task_name = args.task_name
apply_mask = args.applymask
model = args.model
trainset = load_json(args.train_set)["trainset"]
preprocessed = args.preprocessed

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
if preprocessed:
    preprocessed_directory = 'preprocessed'
else:
    preprocessed_directory = 'not_preprocessed'
input_folder = join(prediction_data_root, model, task_name, preprocessed_directory,'input')
output_folder = join(prediction_data_root, model, task_name, preprocessed_directory,'output')
maybe_mkdir_p(input_folder)
maybe_mkdir_p(output_folder)

#%% Putting data to input folder
dataset_preprocessed_folder = join(dataset_folder, preprocessed_directory)
all_cases = subdirs(dataset_preprocessed_folder, join=False)
if preprocessed:
    predicted_cases = []
    print(trainset)
    for case in all_cases:
        print(case)
        if case not in trainset:
            predicted_cases.append(case)
    print(predicted_cases)

else:
    predicted_cases = []
    print(trainset)
    for case in all_cases:
        print(case)
        correct_file = glob(join(dataset_preprocessed_folder, case, "*M*PRAGE_6*.nii.gz"))
        if case not in trainset and len(correct_file) == 1:
            predicted_cases.append(case)
    print(predicted_cases)




for i in range(len(predicted_cases)):
    subject = predicted_cases[i]
    ID, unique_name = subject, subject #TODO
    
    if preprocessed:
        im_file_name = unique_name + "_T1w_7T_Preproc.nii.gz"
        raw_data_file = join(dataset_preprocessed_folder, unique_name, im_file_name)
    else:
        correct_file = glob(join(dataset_preprocessed_folder, subject, "*M*PRAGE_6*.nii.gz"))
        raw_data_file = correct_file[0]

    input_image_file = join(input_folder, ID) # do not specify a file ending! This will be done for you
    input_image_file = input_image_file + "_%04.0d.nii.gz" % 0 # for now, end of file is 0000 because there is only one modality

    image_data, img_obj = read_nifti(raw_data_file)
    if apply_mask:
        # Reading the brain mask
        brain_mask_file_name = unique_name + "_T1w_7T_Preproc_BrainMask.nii.gz"
        brain_mask_file = join(dataset_preprocessed_folder, unique_name, brain_mask_file_name)
        brain_mask_data, brain_mask_obj = read_nifti(brain_mask_file)
        save_nifti(np.multiply(image_data, brain_mask_data[:,:,:,0]), input_image_file, img_obj)
    else:
        save_nifti(image_data, input_image_file, img_obj)
