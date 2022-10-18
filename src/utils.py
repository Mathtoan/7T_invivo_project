import nibabel as nib

def format_time(t):
    ms = int((t - int(t))*100)
    mins, s = divmod(int(t), 60)
    h, m = divmod(mins, 60)
    if h:
        return f'{h:d}:{m:02d}:{s:02d}.{ms:03d}'
    else:
        return f'{m:02d}:{s:02d}.{ms:03d}'

# Read and write nifti functions
def read_nifti(filepath_image):

    img = nib.load(filepath_image)
    image_data = img.get_fdata()

    return image_data, img

def save_nifti(image, filepath_name, img_obj):

    img = nib.Nifti1Image(image, img_obj.affine, header=img_obj.header)
    nib.save(img, filepath_name)