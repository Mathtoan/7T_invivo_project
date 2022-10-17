import os
import argparse
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p, subdirs

# Code
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset_folder', type=str, default='/data/mtduong/7T_invivo_project/dataset/picsl-data',
                    help='Folder where the data to predict are.')
parser.add_argument('-r', '--prediction_data_root', type=str, default='/data/mtduong/7T_invivo_project/nnUNet_prediction')
parser.add_argument('-m', '--model', default="3d_fullres",
                    help="2d, 3d_lowres, 3d_fullres or 3d_cascade_fullres. Default: 3d_fullres")
parser.add_argument('-t', '--task_name', type=str, required=True,
                    help='task name or task ID, required.')
parser.add_argument('-o', '--operation_type', type=str, required=True,
                    help='task name or task ID, required.')
args = parser.parse_args()


prediction_data_root=args.prediction_data_root
dataset_folder=args.dataset_folder
model = args.model
task_name = args.task_name
operation_type = args.operation_type # 'inv1_div_inv2' 'inv2_div_inv1' 'inv2_mul_mp2rage'

input_folder = os.path.join(prediction_data_root, model, task_name, 'picsl-data', operation_type,'input')
output_folder = os.path.join(prediction_data_root, model, task_name, 'picsl-data', operation_type,'output')
maybe_mkdir_p(input_folder)
maybe_mkdir_p(output_folder)

folder_list = subdirs(dataset_folder, join=False)
subjects_list = [i for i in folder_list if i.isdigit()] # Getting only folder with digit for subject id [to be improved]

print("Data preparation for PICSL data")
print(f"input path : {input_folder}")
print(f"output path : {output_folder}")

for subject in subjects_list:
    print(f'Subject {subject}', end='...')
    date = subdirs(os.path.join(dataset_folder, subject), join=False)[-1]
    suffix = f'{date}_{subject}'
    img_fname = f'{suffix}_{operation_type}.nii.gz'
    img_path = os.path.join(dataset_folder, subject, date, img_fname)

    ln_fname = f'{suffix}_{operation_type}_0000.nii.gz'
    ln_path = os.path.join(input_folder, ln_fname)

    if os.path.exists(img_path) and not os.path.exists(ln_path):
        print(f"creating the link {ln_fname}")
        os.symlink(img_path, ln_path)
    else:
        print("link already created")
