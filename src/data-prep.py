from enum import unique
import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data
import argparse

#%% Parser
parser = argparse.ArgumentParser(description='Data preparation for nnUNet training.')
parser.add_argument('-b', '--base', type=str, default=nnUNet_raw_data,
                    help='Set the path of nnUNet_raw_data, default is the one set in environment')
parser.add_argument('-t', '--taskname', type=str, required=True,
                    help='Name of the task, should be in this format : Task[#]_[name]')
parser.add_argument('-l', '--labels', type=str, required=True,
                    help='Path to labels.json')
parser.add_argument('-d', '--dataset', type=str, required=True,
                    help='Path to the dataset')
parser.add_argument('-m', '--applymask', action='store_true',
                    help='Apply mask to image')
parser.add_argument('-i', '--idprefix', type=str,
                    help='Set an ID prefix to file name')

args = parser.parse_args()

#%% Parameter
base = args.base
task_name = args.taskname
labels = load_json(args.labels)
dataset = args.dataset
applymask = args.applymask
idprefix = args.idprefix

config = {}
config['base'] = base
config['task_name'] = task_name
config['labels'] = labels
config['dataset'] = dataset
config['applymask'] = applymask
config['idprefix'] = idprefix

root = '/home/mtduong/7T_invivo_project/'
task_root = join(root, 'task', task_name)
maybe_mkdir_p(task_root)
# maybe_mkdir_p(join(root), )

#%% Read and write nifti functions
def read_nifti(filepath_image):

    img = nib.load(filepath_image)
    image_data = img.get_fdata()

    return image_data, img

def save_nifti(image, filepath_name, img_obj):

    img = nib.Nifti1Image(image, img_obj.affine, header=img_obj.header)
    nib.save(img, filepath_name)

#%% Setting up path in nnUNet_raw_data
target_base = join(base, task_name)
target_imagesTr = join(target_base, "imagesTr")
target_imagesTs = join(target_base, "imagesTs")
target_labelsTs = join(target_base, "labelsTs")
target_labelsTr = join(target_base, "labelsTr")

maybe_mkdir_p(target_imagesTr)
maybe_mkdir_p(target_labelsTs)
maybe_mkdir_p(target_imagesTs)
maybe_mkdir_p(target_labelsTr)

#%% Putting data in nnUNet_raw_data

# Getting data from dataset
training_cases = subdirs(dataset, join=False)
print(training_cases, len(training_cases))

# Assigned an ID to each file for better clarity

if idprefix is not None:
    zfill_number = int(np.log10(len(training_cases)))+1
    IDfile = {}

for i in range(len(training_cases)):
    if idprefix is not None:
        ID = idprefix + '_' + str(i+1).zfill(zfill_number)

        IDfile[ID] = training_cases[i]
    else :
        ID = training_cases[i]
    training_cases[i] = [ID, training_cases[i]]

# Saving ID/unique name into a json file
if idprefix is not None:
    save_json(IDfile, join(task_root, 'ID.json'))


for subject in training_cases:
    ID, unique_name = subject[0], subject-[1]
    print(ID)
    im_file_name = unique_name + "_T1w_7T_Preproc.nii.gz"
    seg_file_name = unique_name + "_3TSegTo7TDeformed.nii.gz"


    input_image_file = join(dataset, unique_name, im_file_name)
    input_segmentation_file = join(dataset, unique_name, seg_file_name)

    output_image_file = join(target_imagesTr, ID)  # do not specify a file ending! This will be done for you
    output_seg_file = join(target_labelsTr, ID)  # do not specify a file ending! This will be done for you

    output_image_file = output_image_file + "_%04.0d.nii.gz" % 0
    output_seg_file = output_seg_file + ".nii.gz"

    # read image, apply mask and save it
    image_data, img_obj = read_nifti(input_image_file)

    # Applying the mask
    if applymask:
        # Reading the brain mask
        brain_mask_file_name = unique_name + "_T1w_7T_Preproc_BrainMask.nii.gz"
        brain_mask_file = join(dataset, unique_name, brain_mask_file_name)
        brain_mask_data, brain_mask_obj = read_nifti(brain_mask_file)

        save_nifti(np.multiply(image_data, brain_mask_data[:,:,:,0]), output_image_file, img_obj)
    
    else:
        save_nifti(image_data, output_image_file, img_obj)

    # read segmentation and save it
    image_data, img_obj = read_nifti(input_segmentation_file)
    save_nifti(image_data, output_seg_file, img_obj)

# finally we can call the utility for generating a dataset.json
generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('MRI',),
                        labels=labels, dataset_name=task_name, license='hands off!')

save_json(config, join(task_root, 'config.json'))

