import os
import numpy as np
import pandas as pd

from batchgenerators.utilities.file_and_folder_operations import subdirs, maybe_mkdir_p

def read_coord(coord):
    ret = coord[1:-1].split(', ')
    ret = np.array([int(i) for i in ret])
    
    return ret

def filter_df(df, dist_threshold, output_path_root=None):
    data_filtered = {}
    for key in df.columns:
        data_filtered[key] = []
    
    df_filtered = pd.DataFrame(data_filtered)
    
    count = 0
    for i in range(len(df)):
        if i == 0:
            add_check= True
        else:
            coord_to_add = read_coord(df['coord'][i])
            for k in range(len(df_filtered)):
                add_check = True
                coord_to_check_with = read_coord(df_filtered["coord"][k])
                
                dist = np.sqrt(np.sum((coord_to_check_with-coord_to_add)**2))
                
                if dist < dist_threshold:
                    add_check = False
                    break
        
        
        if add_check:
            count += 1
            row = df.iloc[i]
            df_filtered = df_filtered.append(row, ignore_index=True)
            
            if not(output_path_root is None) and count <= 3:
                decompose_path = row["name"].split("/")
                decompose_basename = decompose_path[-1].split("_")
                print(decompose_path)
                path = "/"+decompose_path[1]
                for j in range(2, len(decompose_path)-1):
                    if j == 8 :
                        path = os.path.join(path, decompose_path[9])
                    elif j == 9 :
                        path = os.path.join(path, decompose_path[8])
                    else:
                        path = os.path.join(path, decompose_path[j])
                
                output_path = os.path.join(output_path_root, decompose_path[-4]) # TODO : order
                
                maybe_mkdir_p(output_path)
                
                scan_basename = decompose_basename[0]
                seg_basename = decompose_basename[0]
                val_basename = decompose_basename[0]
            
                for j in range(1, len(decompose_basename)):
                    
                    if j != 2:
                        scan_basename += "_" + decompose_basename[j]
                        seg_basename += "_" + decompose_basename[j]
                        val_basename += "_" + decompose_basename[j]
                    else:
                        scan_basename += "_scan"
                        seg_basename += "_seg"
                        val_basename += "_validation"
                
                os.symlink(os.path.join(path, scan_basename), os.path.join(output_path, scan_basename))
                os.symlink(os.path.join(path, seg_basename), os.path.join(output_path, seg_basename))
                os.symlink(os.path.join(path, val_basename), os.path.join(output_path, val_basename))
                    
                print(path)
    
    return df_filtered

cubic_size = 64
dist = cubic_size * np.sqrt(3)

output_root_path = f"/home/mtduong/7T_invivo_project/data/dataset/patches/7T_bis/{cubic_size}x{cubic_size}x{cubic_size}"
output_selected = f"/home/mtduong/7T_invivo_project/data/dataset/patches/selected/7T/{cubic_size}x{cubic_size}x{cubic_size}"
subjects = subdirs(output_root_path, join=False)

for i in range(len(subjects)):
    working_directory = os.path.join(output_root_path, subjects[i])
    
    df = pd.read_csv(os.path.join(working_directory, "overlap.csv"))
    df_right_sorted = df[df['side'] == 'right'].sort_values('mean_overlap', ignore_index=True)
    df_left_sorted = df[df['side'] == 'left'].sort_values('mean_overlap', ignore_index=True)
    
    df_right_sorted.to_csv(os.path.join(working_directory, "overlap_right_sorted.csv"), index=False)
    df_left_sorted.to_csv(os.path.join(working_directory, "overlap_left_sorted.csv"), index=False)

    # print(df_right_sorted['coord'][1])
    df_right_filtered = filter_df(df_right_sorted, dist, output_path_root=output_selected)
    df_left_filtered = filter_df(df_left_sorted, dist, output_path_root=output_selected)
    
    # print('r', len(df_right_sorted), len(df_right_filtered))
    # print('l', len(df_left_sorted), len(df_left_filtered))
    # print(df_right_filtered)

    df_right_filtered.to_csv(os.path.join(working_directory, "overlap_right_filtered.csv"), index=False)
    df_left_filtered.to_csv(os.path.join(working_directory, "overlap_left_filtered.csv"), index=False)