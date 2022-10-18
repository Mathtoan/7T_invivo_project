#%% Imports
import argparse
import os
import time

import nibabel as nib
import numpy as np

from batchgenerators.utilities.file_and_folder_operations import subdirs
from scipy.signal import medfilt

#%% Functions
def format_time(t):
    ms = int((t - int(t))*100)
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return f'{h:d}:{m:02d}:{s:02d}.{ms:03d}'
    else:
        return f'{m:02d}:{s:02d}.{ms:03d}'

# From https://github.com/allucas/mp2rage_functions
def remove_mp2rage_bg(inv1_name, inv2_name, uni_name, output):
    inv1 = nib.load(inv1_name).get_fdata()
    inv2 = nib.load(inv2_name).get_fdata()
    uni = nib.load(uni_name).get_fdata()

    beta=(np.mean(uni[-30:-10,-30:-10,-30:-10])*200)
    INV1final = inv1
    uni_fixed = np.real(((INV1final)*inv2-beta) / (np.abs(INV1final)**2 + np.abs(inv2)**2 + 2*beta))

    uni_fixed_mask = -uni_fixed

    uni_fixed_mask[uni_fixed>0] = -uni_fixed[uni_fixed>0]*(-1)

    uni_fixed_mask = medfilt(uni_fixed_mask, 5)

    uni_fixed_new = uni*(uni_fixed_mask<0.45)

    uni_fixed_nifti = nib.Nifti1Image(uni_fixed_new, nib.load(uni_name).affine)
    nib.save(uni_fixed_nifti, output)

# Code
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_folder', type=str, default='/data/mtduong/7T_invivo_project/dataset/picsl-data')

args = parser.parse_args()

dataset_folder=args.dataset_folder

folder_list = subdirs(dataset_folder, join=False)
subjects_list = [i for i in folder_list if i.isdigit()] # Getting only folder with digit for subject id [to be improved]

print("MP2RAGE Masking PICSL data")

for subject in subjects_list:
    print(f'Subject {subject}')
    dates_list = subdirs(os.path.join(dataset_folder, subject), join=False)
    for date in dates_list:
        suffix = f'{date}_{subject}'
        inv1_path = os.path.join(dataset_folder, subject, date, f'{suffix}_mp2rinv1.nii.gz')
        inv2_path = os.path.join(dataset_folder, subject, date, f'{suffix}_mp2rinv2.nii.gz')
        mp2rage_path = os.path.join(dataset_folder, subject, date, f'{suffix}_mp2rage.nii.gz')
        
        mp2rage_remove_bg_path = os.path.join(dataset_folder, subject, date, f'{suffix}_mp2rage_remove_bg.nii.gz')

        if os.path.exists(inv1_path) and os.path.exists(inv2_path) and os.path.exists(mp2rage_path):
            print("All files found")
            if not os.path.exists(mp2rage_remove_bg_path):
                print(f"creating {mp2rage_remove_bg_path}...", end='', flush=True)
                t = time.time()
                remove_mp2rage_bg(inv1_path, inv2_path, mp2rage_path, mp2rage_remove_bg_path)
                t = time.time() - t
                print(f"done. ({format_time(t)})")
            else:
                print(f"{mp2rage_remove_bg_path} already created")
        else:
            for filepath in (inv1_path, inv2_path, mp2rage_path):
                if os.path.exists(filepath):
                    print(f'{filepath} found')
                else:
                    print(f'{filepath} missing.')
