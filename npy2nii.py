import SimpleITK as sitk
import os
import numpy as np
from tqdm import trange


def get_nii_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.npy':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


if __name__ == '__main__':
    nii_path = r'F:\my_code\copd_PRM\pytorch-CycleGAN-and-pix2pix\save_npy\ground_truth\e'
    npy_path = r'F:\my_code\copd_PRM\pytorch-CycleGAN-and-pix2pix\save_npy\cycle_gan_basic_resnet_9blocks\fakeB'
    save_path = r'F:\my_code\copd_PRM\pytorch-CycleGAN-and-pix2pix\save_npy\cycle_gan_basic_resnet_9blocks\fakeB_nii'
    nii_list = get_nii_listdir(nii_path)
    nii_list.sort()
    img_list = get_listdir(npy_path)
    img_list.sort()

    k = 0
    for nii in nii_list:
        sitk_img = sitk.ReadImage(nii)
        img_arr = sitk.GetArrayFromImage(sitk_img)

        new_img = np.zeros_like(img_arr)
        for i in trange(img_arr.shape[0]):
            image = np.load(img_list[k + i])
            new_img[i, :, :] = image
        k += img_arr.shape[0]
        new_img = sitk.GetImageFromArray(new_img)
        new_img.SetDirection(sitk_img.GetDirection())
        new_img.SetSpacing(sitk_img.GetSpacing())
        new_img.SetOrigin(sitk_img.GetOrigin())
        _, fullflname = os.path.split(nii)
        sitk.WriteImage(new_img, os.path.join(save_path, fullflname))
