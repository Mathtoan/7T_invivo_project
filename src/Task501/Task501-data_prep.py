
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data, preprocessing_output_dir
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import os.path
import glob
import warnings
import shutil
import random
from scipy import ndimage
import SimpleITK as sitk
import nibabel as nib
import importlib
import torch
from scipy.ndimage import rotate, map_coordinates, gaussian_filter
from scipy.ndimage.filters import convolve
from skimage.filters import gaussian
from skimage.segmentation import find_boundaries
from torchvision.transforms import Compose
from nipype.interfaces.ants import N4BiasFieldCorrection
import scipy.ndimage as ndi
from scipy.ndimage import map_coordinates, gaussian_filter
from typing import Tuple, List, Union

def read_nifti(filepath_image):

    img = nib.load(filepath_image)
    image_data = img.get_fdata()

    return image_data, img


def save_nifti(image, filepath_name, img_obj):

    img = nib.Nifti1Image(image, img_obj.affine, header=img_obj.header)
    nib.save(img, filepath_name)


if __name__ == '__main__':
    """
    nnU-Net was originally built for 3D images. It is also strongest when applied to 3D segmentation problems because a
    large proportion of its design choices were built with 3D in mind. Also note that many 2D segmentation problems,
    especially in the non-biomedical domain, may benefit from pretrained network architectures which nnU-Net does not
    support.
    Still, there is certainly a need for an out of the box segmentation solution for 2D segmentation problems. And
    also on 2D segmentation tasks nnU-Net cam perform extremely well! We have, for example, won a 2D task in the cell
    tracking challenge with nnU-Net (see our Nature Methods paper) and we have also successfully applied nnU-Net to
    histopathological segmentation problems.
    Working with 2D data in nnU-Net requires a small workaround in the creation of the dataset. Essentially, all images
    must be converted to pseudo 3D images (so an image with shape (X, Y) needs to be converted to an image with shape
    (1, X, Y). The resulting image must be saved in nifti format. Hereby it is important to set the spacing of the
    first axis (the one with shape 1) to a value larger than the others. If you are working with niftis anyways, then
    doing this should be easy for you. This example here is intended for demonstrating how nnU-Net can be used with
    'regular' 2D images. We selected the massachusetts road segmentation dataset for this because it can be obtained
    easily, it comes with a good amount of training cases but is still not too large to be difficult to handle.
    """

    # download dataset from https://www.kaggle.com/insaff/massachusetts-roads-dataset
    # extract the zip file, then set the following path according to your system:
    ## PK
    #base = '/Users/pulkit/Desktop/invivoUnetData'
    #base = '/Users/pulkit/Desktop/invivoUnetData/testFeb9th'
    #base = '/Users/pulkit/Desktop/invivoUnetData/coronal'

    # base = '/Users/pulkit/Desktop/ADNI_WMH_Masks/'

    base = '/data/mtduong/nnUNet/nnUNet_raw_data_base/nnUNet_raw_data'
    task_name = 'Task501_7Tgm'
    labels = {0: "Background", 1: 'Label 1', 2: 'Label 2', 3: 'Label 3',
              4: 'Label 4', 5: 'Label 5', 6: 'Label 6'}

    #base = '/media/fabian/data/road_segmentation_ideal'
    # this folder should have the training and testing subfolders

    # now start the conversion to nnU-Net:
    target_base = join(base, task_name)
    target_imagesTr = join(target_base, "imagesTr")
    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")
    target_labelsTr = join(target_base, "labelsTr")

    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    dataset = '/home/mtduong/7T_invivo_project/data/warpSeg'
    training_cases = subdirs(dataset, join=False)

    print(training_cases, len(training_cases))

    # #### get the list of the input images for this fold

    for unique_name in training_cases:
        print(unique_name)
        im_file_name = unique_name + "_T1w_7T_Preproc.nii.gz"
        seg_file_name = unique_name + "_3TSegTo7TDeformed.nii.gz"

        input_image_file = join(dataset, unique_name, im_file_name)
        input_segmentation_file = join(dataset, unique_name, seg_file_name)

        output_image_file = join(target_imagesTr, unique_name)  # do not specify a file ending! This will be done for you
        output_image_file = output_image_file + "_%04.0d.nii.gz" % 0

        output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = output_seg_file + ".nii.gz"

        # read image and save it
        image_data, img_obj = read_nifti(input_image_file)
        save_nifti(image_data, output_image_file, img_obj)

        # read segmentation and save it
        image_data, img_obj = read_nifti(input_segmentation_file)
        save_nifti(image_data, output_seg_file, img_obj)

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('MRI',),
                          labels=labels, dataset_name=task_name, license='hands off!')

    # """
    # once this is completed, you can use the dataset like any other nnU-Net dataset. Note that since this is a 2D
    # dataset there is no need to run preprocessing for 3D U-Nets. You should therefore run the
    # `nnUNet_plan_and_preprocess` command like this:

    # > nnUNet_plan_and_preprocess -t 120 -pl3d None

    # once that is completed, you can run the trainings as follows:
    # > nnUNet_train 2d nnUNetTrainerV2 120 FOLD

    # (where fold is again 0, 1, 2, 3 and 4 - 5-fold cross validation)

    # there is no need to run nnUNet_find_best_configuration because there is only one model to shoose from.
    # Note that without running nnUNet_find_best_configuration, nnU-Net will not have determined a postprocessing
    # for the whole cross-validation. Spoiler: it will determine not to run postprocessing anyways. If you are using
    # a different 2D dataset, you can make nnU-Net determine the postprocessing by using the
    # `nnUNet_determine_postprocessing` command
    # """
