import SimpleITK as sitk
import os
import copy
import numpy as np
import tqdm


def get_listdir(path):  # 获取目录下所有png格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def PRM(img_i, img_e, save_path):
    sitk_img_i = sitk.ReadImage(img_i)
    i_img_arr = sitk.GetArrayFromImage(sitk_img_i)
    sitk_img_e = sitk.ReadImage(img_e)
    e_img_arr = sitk.GetArrayFromImage(sitk_img_e)

    temp = np.zeros_like(i_img_arr)

    temp[(i_img_arr >= -950) & (e_img_arr >= -856)] = 2
    temp[(i_img_arr >= -950) & (e_img_arr < -856)] = 4
    temp[(i_img_arr < -950) & (e_img_arr >= -856)] = 3
    temp[(i_img_arr < -950) & (e_img_arr < -856)] = 1

    temp[i_img_arr > -500] = 0

    new_img = sitk.GetImageFromArray(temp)
    new_img.SetDirection(sitk_img_i.GetDirection())
    new_img.SetOrigin(sitk_img_i.GetOrigin())
    new_img.SetSpacing(sitk_img_i.GetSpacing())
    _, fullflname = os.path.split(img_i)
    sitk.WriteImage(new_img, os.path.join(save_path, fullflname))


if __name__ == '__main__':
    i_path = r'F:\my_code\copd_PRM\pytorch-CycleGAN-and-pix2pix\save_npy\ground_truth\i'
    e_path = r'F:\my_code\copd_PRM\pytorch-CycleGAN-and-pix2pix\save_npy\cycle_gan_pixel_unet_256\fakeB_nii'
    save_path = r'F:\my_code\copd_PRM\pytorch-CycleGAN-and-pix2pix\save_npy\cycle_gan_pixel_unet_256\fake_PRM'
    i_list = get_listdir(i_path)
    i_list.sort()
    e_list = get_listdir(e_path)
    e_list.sort()
    for i in tqdm.trange(len(i_list)):
        PRM(i_list[i], e_list[i], save_path)
