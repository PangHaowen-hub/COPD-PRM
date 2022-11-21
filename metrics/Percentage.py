import SimpleITK as sitk
import numpy as np
import os
from tqdm import trange
import collections
from scipy import stats


def get_listdir(path):
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def per(mask_path, pred_path):
    mask_sitk_img = sitk.ReadImage(mask_path)
    mask_img_arr = sitk.GetArrayFromImage(mask_sitk_img)
    pred_sitk_img = sitk.ReadImage(pred_path)
    pred_img_arr = sitk.GetArrayFromImage(pred_sitk_img)
    count_mask = collections.Counter(mask_img_arr.flatten())
    count_pred = collections.Counter(pred_img_arr.flatten())
    mask_all = count_mask[1] + count_mask[2] + count_mask[3] + count_mask[4]
    pred_all = count_pred[1] + count_pred[2] + count_pred[3] + count_pred[4]

    mask_per_1 = count_mask[1] / mask_all
    mask_per_2 = count_mask[2] / mask_all
    mask_per_3 = count_mask[3] / mask_all
    mask_per_4 = count_mask[4] / mask_all

    pred_per_1 = count_pred[1] / pred_all
    pred_per_2 = count_pred[2] / pred_all
    pred_per_3 = count_pred[3] / pred_all
    pred_per_4 = count_pred[4] / pred_all
    return [pred_per_1, pred_per_2, pred_per_3, pred_per_4], [mask_per_1, mask_per_2, mask_per_3, mask_per_4]


if __name__ == '__main__':
    mask_path = r'F:\my_code\copd_PRM\pytorch-CycleGAN-and-pix2pix\save_npy\ground_truth\PRM'
    pred_path = r'F:\my_code\copd_PRM\pytorch-CycleGAN-and-pix2pix\save_npy\cycle_gan_pixel_unet_256\fake_PRM'
    mask = get_listdir(mask_path)
    mask.sort()
    pred = get_listdir(pred_path)
    pred.sort()
    pre_list_1 = []
    pre_list_2 = []
    pre_list_3 = []
    pre_list_4 = []

    mask_list_1 = []
    mask_list_2 = []
    mask_list_3 = []
    mask_list_4 = []
    for i in trange(len(mask)):
        count_pred, count_mask = per(mask[i], pred[i])
        pre_list_1.append(count_pred[0])
        pre_list_2.append(count_pred[1])
        pre_list_3.append(count_pred[2])
        pre_list_4.append(count_pred[3])

        mask_list_1.append(count_mask[0])
        mask_list_2.append(count_mask[1])
        mask_list_3.append(count_mask[2])
        mask_list_4.append(count_mask[3])
    print(stats.ttest_rel(pre_list_1, mask_list_1))
    print(stats.ttest_rel(pre_list_2, mask_list_2))
    print(stats.ttest_rel(pre_list_3, mask_list_3))
    print(stats.ttest_rel(pre_list_4, mask_list_4))
