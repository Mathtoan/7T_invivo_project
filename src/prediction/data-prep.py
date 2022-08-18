import numpy as np
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.paths import nnUNet_raw_data
import argparse
from random import shuffle

#%% Parser
parser = argparse.ArgumentParser(description='Data preparation for nnUNet training.')
parser.add_argument('-t', '--taskname', metavar='task name', type=str, required=True,
                    help='Name of the task, should be in this format : Task[#]_[name]')
parser.add_argument('-m', '--applymask', action='store_true',
                    help='Apply mask to image')
parser.add_argument('-i', '--idprefix', metavar='ID prefix', type=str,
                    help='Set an ID prefix to file name')

args = parser.parse_args()

#%% Parameter
task_name = args.taskname
applymask = args.applymask
idprefix = args.idprefix

