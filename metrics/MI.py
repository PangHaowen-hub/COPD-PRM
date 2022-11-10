import numpy as np
import os
from tqdm import trange
import SimpleITK as sitk


def get_listdir(path):  # 获取目录下所有gz格式文件的地址，返回地址list
    tmp_list = []
    for file in os.listdir(path):
        if os.path.splitext(file)[1] == '.gz':
            file_path = os.path.join(path, file)
            tmp_list.append(file_path)
    return tmp_list


def read_image(path):
    img = sitk.ReadImage(path)
    nda = sitk.GetArrayFromImage(img)
    nda[nda < -1000] = -1000
    nda[nda > 0] = 0
    return nda


def hxx(x, y):
    x = np.reshape(x, -1)
    y = np.reshape(y, -1)

    size = x.shape[-1]
    px = np.histogram(x, 1000, (-1000, 0))[0] / size
    py = np.histogram(y, 1000, (-1000, 0))[0] / size
    hx = - np.sum(px * np.log(px + 1e-8))
    hy = - np.sum(py * np.log(py + 1e-8))

    hxy = np.histogram2d(x, y, 1000, [[-1000, 0], [-1000, 0]])[0]
    hxy /= (1.0 * size)
    hxy = - np.sum(hxy * np.log(hxy + 1e-8))

    r = hx + hy - hxy
    return r


if __name__ == '__main__':
    pred_path = r'F:\my_code\NCCT2CECT\pix2pix-2d\pix2pixHD_ncct2cect\pred\SCECT_lungbox_extractlung'
    gt_path = r'H:\CT2CECT\pix2pix\data\cect_a_lungbox_extractlung'
    pred_list = get_listdir(pred_path)
    pred_list.sort()
    gt_list = get_listdir(gt_path)
    gt_list.sort()
    pred_list = pred_list
    gt_list = gt_list[30:]
    mean_MI = []
    for i in trange(len(pred_list)):
        npImg1 = read_image(pred_list[i])
        npImg2 = read_image(gt_list[i])
        mi = hxx(npImg1, npImg2)
        print(mi)
        mean_MI.append(mi)

    print('avg MI: ', np.mean(mean_MI))
    print('std MI: ', np.std(mean_MI))
