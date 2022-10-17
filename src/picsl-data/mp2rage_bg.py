import os
import argparse
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subdirs

# Code
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_folder', type=str, default='/data/mtduong/7T_invivo_project/dataset/picsl-data',
                    help='Folder where the data to predict are.')

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
        inv1 = os.path.join(dataset_folder, subject, date, f'{suffix}_mp2rinv1.nii.gz')
        inv2 = os.path.join(dataset_folder, subject, date, f'{suffix}_mp2rinv2.nii.gz')
        mp2rage = os.path.join(dataset_folder, subject, date, f'{suffix}_mp2rage.nii.gz')
        
        mp2rage_remove_bg = os.path.join(dataset_folder, subject, date, f'{suffix}_mp2rage_remove_bg.nii.gz')

        if os.path.exists(inv1) and os.path.exists(inv2) and os.path.exists(mp2rage):
            print("All files found")
            print(f"creating {mp2rage_remove_bg}", end="...")
            if not os.path.exists(mp2rage_remove_bg):
                os.system(f"python3 ~/mp2rage_functions/code/remove_mp2rage_bg.py -inv1 {inv1} -inv2 {inv2} -uni {mp2rage} -o {mp2rage_remove_bg} ")
                print("done")
            else:
                print("already created")
        exit()
        # inv2 = os.path.join(dataset_folder, subject, date, img_fname)

        # ln_fname = f'{suffix}_{operation_type}_0000.nii.gz'
        # ln_path = os.path.join(input_folder, ln_fname)

        # if os.path.exists(img_path) and not os.path.exists(ln_path):
        #     print(f"creating the link {ln_fname}")
        #     os.symlink(img_path, ln_path)
        # else:
        #     print("link already created")
