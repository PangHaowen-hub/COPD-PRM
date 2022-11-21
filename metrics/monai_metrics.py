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

    img_arr = torch.from_numpy(img_arr)
    img_arr = img_arr.unsqueeze(dim=0).unsqueeze(dim=0)
    gt_arr = torch.from_numpy(gt_arr)
    gt_arr = gt_arr.unsqueeze(dim=0).unsqueeze(dim=0)

    mae = monai.metrics.MAEMetric("none")
    mse = monai.metrics.MSEMetric("none")
    rmse = monai.metrics.RMSEMetric("none")
    psnr = monai.metrics.PSNRMetric(1000)

    return mae(img_arr, gt_arr), mse(img_arr, gt_arr), rmse(img_arr, gt_arr), psnr(img_arr, gt_arr)


if __name__ == '__main__':
    pred_path = r'F:\my_code\copd_PRM\pytorch-CycleGAN-and-pix2pix\save_npy\cycle_gan_basic_SwinUNETR_25epoch\fakeB_nii'
    gt_path = r'F:\my_code\copd_PRM\pytorch-CycleGAN-and-pix2pix\save_npy\ground_truth\e'

    pred_list = get_listdir(pred_path)
    pred_list.sort()
    gt_list = get_listdir(gt_path)
    gt_list.sort()
    pred_list = pred_list
    gt_list = gt_list
    mean_mae = []
    mean_mse = []
    mean_rmse = []
    mean_psnr = []
    for i in trange(min(len(pred_list), len(gt_list))):
        temp = loss(gt_list[i], pred_list[i])
        mean_mae.append(temp[0].item())
        mean_mse.append(temp[1].item())
        mean_rmse.append(temp[2].item())
        mean_psnr.append(temp[3].item())

    print('avg mae: ', np.mean(mean_mae))
    print('std mae: ', np.std(mean_mae))

    print('avg mse: ', np.mean(mean_mse))
    print('std mse: ', np.std(mean_mse))

    print('avg rmse: ', np.mean(mean_rmse))
    print('std rmse: ', np.std(mean_rmse))

    print('avg psnr: ', np.mean(mean_psnr))
    print('std psnr: ', np.std(mean_psnr))
