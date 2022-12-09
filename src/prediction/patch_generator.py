import argparse
import os
import numpy as np
import nibabel as nib

from random import shuffle
from glob import glob

from batchgenerators.utilities.file_and_folder_operations import subdirs, maybe_mkdir_p

# c3d img1.img -region 20x20x20vox 50x60x70vox -o img2.img
# c3d img1.img -region 25% 50% -o img3.img

#%% Functions
def read_nifti(filepath_image):

    img = nib.load(filepath_image)
    image_data = img.get_fdata()

    return image_data, img

def extract_region(fname_input, fname_output, origin, size):
    bash_command = f"c3d {fname_input} -region {origin[0]}x{origin[1]}x{origin[2]}vox {size[0]}x{size[1]}x{size[2]}vox -o {fname_output}"
    os.system(bash_command)

def get_patch_origin(fname_input, label=None, n=None): 
    origin_coord = {}
    
    im_data, _ = read_nifti(fname_input)
    
    labels = np.unique(im_data)
    labels = labels[1:] # Removing bg label
    
    for i in range(len(labels)):
        x, y, z, = np.where(im_data == labels[i])
        
        if len(x) != 0 and int(labels[i]) != 0 :
            idx = [i for i in range(len(x))]
            shuffle(idx)
            
            origin_coord[f"{int(labels[i])}"] = [x[idx[0]]+1, y[idx[0]]+1, z[idx[0]]+1]
        
    
    # x, y, z, = np.where(im_data == label)
    
    # idx = [i for i in range(len(x))]
    # shuffle(idx)
    
    # if n is None:
    #     n = len(x)
    # for i in range(n):
    #     origin_coord.append([x[idx[i]]+1, y[idx[i]]+1, z[idx[i]]+1])
    #     # print([x[idx[i]], y[idx[i]], z[idx[i]]])
    
    return origin_coord

#%% Code
def test():
    from itertools import permutations
    output_root_path = "/home/mtduong/7T_invivo_project/tmp/playground_patch"
    output_dir = os.path.join(output_root_path, "20191217x1212")
    maybe_mkdir_p(output_dir)
    
    antsct_path = "/home/mtduong/data/SC7T/20191217x1212/antsct/"
    dkt_path =  os.path.join(antsct_path, "sub-125678_ses-20191217x1409_DKT31.nii.gz")
    scan_path =  os.path.join(antsct_path, "sub-125678_ses-20191217x1409_PreprocessedInput.nii.gz")
    
    size = (64, 64, 64)
    # origin_coord = get_patch_origin(dkt_path, 2016, n=4)
    origin_coord = get_patch_origin(dkt_path)
    
    i = 1
    for label in origin_coord:
        print(f"{label} ({i}/{len(origin_coord)})")
        extract_region(scan_path, os.path.join(output_dir, f"{label}_scan_{size[0]}x{size[1]}x{size[2]}.nii.gz"), origin_coord[label], size)
        extract_region(dkt_path, os.path.join(output_dir, f"{label}_dkt_{size[0]}x{size[1]}x{size[2]}.nii.gz"), origin_coord[label], size)
        
        i += 1

    # # To check the coordonate 
    # coords = list(permutations([134, 145, 190]))
    
    # im_data, _ = read_nifti(dkt_path)
    
    # for coord in coords:
    #     # print(im_data[coord[0],coord[1],coord[2]])
    #     print(f"label({coord[0]},{coord[1]},{coord[2]}) : {im_data[coord[0],coord[1],coord[2]]}")


def main():
    SC7T_path = "/home/mtduong/data/SC7T/"
    output_root_path = "/home/mtduong/7T_invivo_project/data/dataset/patches"
    maybe_mkdir_p(output_root_path)

    subjects_list = subdirs(SC7T_path, join=False)
    subjects_list = [i for i in subjects_list if os.path.exists(os.path.join(SC7T_path, i, "antsct"))]

    print(subjects_list)

    size = (32, 32, 32) # TODO : have to actually decide the size (to test : 32x32x32)
    str_size = f"{size[0]}x{size[1]}x{size[2]}"
    for k in range(len(subjects_list)):
        subject = subjects_list[k]
        print(f"Subject {subject}")
    
        
        antsct_path = os.path.join(SC7T_path, subject, 'antsct')
        
        if os.path.exists(antsct_path):
            print("antsct dir exists")
            # TODO : fix the 'something_TODO' using glob 
            dkt_path = glob(os.path.join(antsct_path, '*_DKT31.nii.gz'))
            if len(dkt_path) == 1 :
                dkt_path = dkt_path[0]
            else:
                print("DKT file not found or ambigues files")
                continue
            origin_coord = get_patch_origin(dkt_path)
            
            scan_path = glob(os.path.join(antsct_path, '*_PreprocessedInput.nii.gz'))
            if len(scan_path) == 1 :
                scan_path = scan_path[0]
            else:
                print("PreprocessedInput file not found or ambigues files")
                continue
            
            seg_path = glob(os.path.join(antsct_path, '*_BrainSegmentation.nii.gz'))
            if len(seg_path) == 1 :
                seg_path = seg_path[0]
            else:
                print("BrainSegmentation file not found or ambigues files")
                continue
            
            counter = 1
            for label in origin_coord:
                output_dir = os.path.join(output_root_path, subject, str_size, label)
                maybe_mkdir_p(output_dir)
                
                print(f"Label {label} ({counter}/{len(origin_coord)})")
                f_scan_name_output = os.path.join(output_dir, f"{subject}_{label}_scan_{str_size}.nii.gz") 
                f_seg_name_output = os.path.join(output_dir, f"{subject}_{label}_seg_{str_size}.nii.gz") 
                
                extract_region(scan_path, f_scan_name_output, origin_coord[label], size)
                extract_region(seg_path, f_seg_name_output, origin_coord[label], size)
                counter += 1
        else:
            print("antsct dir not found")
            continue
    
    
main()
