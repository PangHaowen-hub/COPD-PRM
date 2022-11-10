import monai.metrics
import SimpleITK as sitk
import torch
from tqdm import trange
import os
import numpy as np


def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def loss(gt_name, pred_name):
    sitk_img = sitk.ReadImage(gt_name)
    img_arr = sitk.GetArrayFromImage(sitk_img)
    sitk_gt = sitk.ReadImage(pred_name)
    gt_arr = sitk.GetArrayFromImage(sitk_gt)

    img_arr[img_arr > 0] = 0
    img_arr[img_arr < -1000] = -1000
    gt_arr[gt_arr > 0] = 0
    gt_arr[gt_arr < -1000] = -1000

    img_arr = (img_arr + 1000) / 1000
    gt_arr = (gt_arr + 1000) / 1000

    img_arr = torch.from_numpy(img_arr)
    img_arr = img_arr.unsqueeze(dim=0).unsqueeze(dim=0)
    gt_arr = torch.from_numpy(gt_arr)
    gt_arr = gt_arr.unsqueeze(dim=0).unsqueeze(dim=0)

    img_arr = img_arr.type(torch.float32)
    gt_arr = gt_arr.type(torch.float32)

    data_range = img_arr.max().unsqueeze(0) - img_arr.min().unsqueeze(0)
    ssim = monai.metrics.regression.SSIMMetric(data_range=data_range, spatial_dims=3)._compute_metric(img_arr, gt_arr)

    return ssim


if __name__ == '__main__':
    pred_path = r'H:\CT2CECT\pix2pix\data\registration_ncct2cect_a_lungbox'
    gt_path = r'H:\CT2CECT\pix2pix\data\cect_a_lungbox'

    pred_list = get_listdir(pred_path)
    pred_list.sort()
    gt_list = get_listdir(gt_path)
    gt_list.sort()
    pred_list = pred_list
    gt_list = gt_list

    mean_ssim = []

    for i in trange(min(len(pred_list), len(gt_list))):
        temp = loss(gt_list[i], pred_list[i])
        mean_ssim.append(temp.item())

    print('avg ssim: ', np.mean(mean_ssim))
    print('std ssim: ', np.std(mean_ssim))
