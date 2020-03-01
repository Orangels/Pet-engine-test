import os
import sys
import shutil
import argparse

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from PIL import Image


sys.path.append('/home/user/workspace/priv-0220/Pet-engine')
from core.semseg_priv_config import cfg_priv
sys.path.append(cfg_priv.PET_ROOT)
from modules import pet_engine

# System libs
import time
import argparse
from scipy.io import loadmat


def main():
    print(pet_engine.MODULES.keys())
    module = pet_engine.MODULES['Semantc_Segmentation']
    # img = Image.open('/home/user/workspace/priv-0220/privision_test/P000014.png').convert('RGB')
    dir_result = '/home/user/workspace/priv-0220/privision_test'
    gpu_id = cfg_priv.MODULES.SEMSEG.GPU_ID
    torch.cuda.set_device(gpu_id)


    # opencv 和 PIL 的区别
    semseg_inference = module(cfg_file='/home/user/workspace/priv-0220/Vas/yaml/seg_smart_ground.yaml')
    sum = 0
    for i in range(1):
        time_cv_start = time.time()
        # img_cv = cv2.imread('/home/user/workspace/priv-0220/privision_test/P000014.png')
        # img_cv = img_cv[:, :, ::-1]
        # # img_cv = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        time_PIL_start = time.time()
        img = Image.open('/home/user/workspace/priv-0220/privision_test/P000014.png').convert('RGB')
        #
        # print(type(img_cv))
        # print(type(img))
        #
        print('cv time cost {}'.format(time_PIL_start-time_cv_start))
        print('****')
        print('PIL time cost {}'.format(time.time()-time_PIL_start))
        pred = semseg_inference(img)
        # np.save('P000014_engine', pred)
        # print('total time cost {}'.format(time.time() - time_cv_start))
        sum += time.time() - time_cv_start

    print('arvg time cost {}'.format(sum/50))
    # time_save_start = time.time()
    # # np.save('mask', im_vis)
    # time_save_end = time.time()
    # print('save cost {}'.format(time_save_end-time_save_start))
    #
    # time_load_start = time.time()
    # aaa = np.load('mask.npy')
    # time_load_end = time.time()
    # print('load cost {}'.format(time_load_end - time_load_start))

    # print(im_vis.dtype)
    # print(im_vis.shape)

    # Main loop
    # im_vis = semseg_inference(img)
    if cfg_priv.MODULES.SEMSEG.TEST.VIS_ENABLED:
        Image.fromarray(im_vis).save(os.path.join(dir_result, 'smart_ground14.png'))
    # print(im_vis)
    print('Evaluation Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Validation"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    args = parser.parse_args()

    main()
