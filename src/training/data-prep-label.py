import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data
import argparse
from random import shuffle

#%% Parser
def percentFloat (string):
    value = float(string)
    if value < 0. or value > 1.:
        raise argparse.ArgumentTypeError('Value has to be between 0 and 1')
    return value

parser = argparse.ArgumentParser(description='Data preparation for nnUNet training.')
parser.add_argument('-b', '--base', metavar='PATH', type=str, default=nnUNet_raw_data,
                    help='Set the path of nnUNet_raw_data, default is the one set in environment')
parser.add_argument('-t', '--taskname', metavar='task name', type=str, required=True,
                    help='Name of the task, should be in this format : Task[#]_[name]')
parser.add_argument('-l', '--labels', metavar='PATH', type=str, default='/home/mtduong/7T_invivo_project/labels.json',
                    help='Path to labels.json')
parser.add_argument('-d', '--dataset', metavar='PATH', type=str, default='/data/mtduong/7T_invivo_project/dataset/preprocessed',
                    help='Path to the dataset')
parser.add_argument('-m', '--applymask', action='store_true',
                    help='Apply mask to image')
parser.add_argument('-i', '--idprefix', metavar='ID prefix', type=str,
                    help='Set an ID prefix to file name')
parser.add_argument('-p', '--train_percentage', metavar='percentage', type=percentFloat, default=1.,
                    help='Percentage of train subject in the dataset')
parser.add_argument('-n', '--dry_run', action='store_true',
                    help="Doing a dry run by not copying the file to nnUNet_raw_data."
                         "For debug purposes")
parser.add_argument('-r','--labels_to_remove', nargs='+', required=True,
                    help='Labels to remove')
parser.add_argument('-R','--reference_label', type=int, required=True,
                    help='Reference label for merging')
parser.add_argument('-Rm', '--reference_name', type=str, default=None,
                    help='New name of reference')
parser.add_argument('-M','--labels_to_merge', nargs='+', required=True,
                    help='Labels to remove')
parser.add_argument('-tr', '--train_set', default="/data/mtduong/7T_invivo_project/dataset/trainset1.json",
                    help="path of the train set json file.")


args = parser.parse_args()

#%% Parameter configutation
base = args.base
task_name = args.taskname
labels = load_json(args.labels)
dataset = args.dataset
applymask = args.applymask
idprefix = args.idprefix
train_percentage = args.train_percentage
dry_run = args.dry_run
labels_to_remove = args.labels_to_remove
reference_label = args.reference_label
reference_name = args.reference_name
if reference_name is None:
    reference_name = labels[str(reference_label)]
labels_to_merge = args.labels_to_merge
trainset = load_json(args.train_set)["trainset"]

#%% Labels management
# Checking labels removing and merging
if str(reference_label) in labels_to_remove:
    raise ValueError("reference label is a label to be removed")

for x in labels_to_merge:
    if x in labels_to_remove:
        raise ValueError("label to be merged is a label to be removed")

# Removing label to be removed in label list
print("removing label", [(x, labels[x]) for x in labels_to_remove])
for i in range(len(labels_to_remove)):
    del labels[labels_to_remove[i]]

# Removing label to merge from label list
print("merging labels", [(x, labels[x]) for x in labels_to_merge], "into", (reference_label, labels[str(reference_label)]), 
      "refered as", reference_name)
labels_used_during_merging = [int(x) for x in labels_to_merge]
labels_used_during_merging.append(reference_label)
labels_used_during_merging.sort()
min_label_merged = labels_used_during_merging[0]

for i in range(len(labels_used_during_merging)):
    if i == 0:
        labels[str(labels_used_during_merging[i])] = reference_name
    else:
        del labels[str(labels_used_during_merging[i])]

print("Labels are", labels)

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

if not dry_run:
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

#%% Putting data in nnUNet_raw_data

# Getting data from dataset
all_cases = subdirs(dataset, join=False)
training_cases = []
print(trainset)
for case in all_cases:
    print(case)
    if case in trainset:
        training_cases.append(case)
print(training_cases)
num_subjects = len(training_cases)

# Assigned an ID to each file for better clarity
if idprefix is not None:
    zfill_number = int(np.log10(num_subjects))+1
    IDfile = {}

for i in range(num_subjects):
    if idprefix is not None:
        ID = idprefix + '_' + str(i+1).zfill(zfill_number)

        IDfile[ID] = training_cases[i]
    else :
        ID = training_cases[i]
    training_cases[i] = [ID, training_cases[i]]

# Train / test repartition
num_train_subjects = round(num_subjects*train_percentage)
train_test_set = ["train" if i<num_train_subjects else "test" for i in range(num_subjects)]
shuffle(train_test_set)

for i in range(num_subjects):
    subject = training_cases[i]
    ID, unique_name = subject[0], subject[1]
    print(ID, train_test_set[i])
    im_file_name = unique_name + "_T1w_7T_Preproc.nii.gz"
    seg_file_name = unique_name + "_3TSegTo7TDeformed.nii.gz"

    input_image_file = join(dataset, unique_name, im_file_name)
    input_segmentation_file = join(dataset, unique_name, seg_file_name)

    if train_test_set[i] == "train":
        target_images, target_labels = target_imagesTr, target_labelsTr
    elif train_test_set[i] == "test":
        target_images, target_labels = target_imagesTs, target_labelsTs

    output_image_file = join(target_images, ID)  # do not specify a file ending! This will be done for you
    output_seg_file = join(target_labels, ID)  # do not specify a file ending! This will be done for you

    output_image_file = output_image_file + "_%04.0d.nii.gz" % 0
    output_seg_file = output_seg_file + ".nii.gz"

    # read image, apply mask and save it
    if not dry_run:
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

        # Removing labels
        for i in range(len(labels_to_remove)):
            image_data[image_data==int(labels_to_remove[i])] = 0
        
        # Merging labels
        for i in range(len(labels_to_merge)):
            image_data[image_data==int(labels_to_merge[i])] = int(min_label_merged)
        save_nifti(image_data, output_seg_file, img_obj)

#%% setting json files
# finally we can call the utility for generating a dataset.json
if not dry_run:
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('MRI',),
                            labels=labels, dataset_name=task_name, license='hands off!')

# Saving ID/unique name into a json file
if not dry_run and idprefix is not None:
    save_json(IDfile, join(task_root, 'ID.json'))

# Saving configuration in json file
if not dry_run:
    config = {}
    config['base'] = base
    config['task_name'] = task_name
    config['labels'] = labels
    config['dataset'] = dataset
    config['applymask'] = applymask
    config['idprefix'] = idprefix
    config['num_subjects'] = num_subjects
    config['num_train_subjects'] = num_train_subjects
    config['num_test_subjects'] = num_subjects - num_train_subjects
    config['trainset'] = trainset

    save_json(config, join(task_root, 'config.json'))

