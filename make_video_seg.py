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

from scipy.io import loadmat

colors = loadmat('/home/user/workspace/priv-0220/Pet-engine/utils/color150.mat')['colors']

confidence = 0.4

mode_type_car = 'car'
mode_type_fog = 'fog'
mode_type = [mode_type_car, mode_type_fog]
mode_index = 0

video_path = '/home/user/Program/ls/video_test/20200226_123648_Trim.mp4'
# video_path = '/home/user/Program/ls/video_test/20200226_123648_Trim_{}_test.mp4'.format(mode_type[abs(mode_index-1)])
out_path = '/home/user/Program/ls/video_test/20200226_123648_Trim_seg.mp4'
img_path = '/home/user/workspace/priv-0220/privision_test/video_imgs'
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')  # opencv3.0
# fourcc = cv2.VideoWriter_fourcc('X','V','I','D')  # opencv3.0
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # 保存 mp4

print(pet_engine.MODULES.keys())
module = pet_engine.MODULES['Semantc_Segmentation']
gpu_id = cfg_priv.MODULES.SEMSEG.GPU_ID
torch.cuda.set_device(gpu_id)
semseg_inference = module(cfg_file='/home/user/workspace/priv-0220/Vas/yaml/seg_smart_ground.yaml')


def color_encode(labelmap, colors, mode='RGB'):
    labelmap = labelmap.astype('int')
    labelmap_rgb = np.zeros((labelmap.shape[0], labelmap.shape[1], 3),
                            dtype=np.uint8)
    for label in np.unique(labelmap):
        if label < 0:
            continue
        labelmap_rgb += (labelmap == label)[:, :, np.newaxis] * \
                        np.tile(colors[label],
                                (labelmap.shape[0], labelmap.shape[1], 1))

    if mode == 'BGR':
        return labelmap_rgb[:, :, ::-1]
    else:
        return labelmap_rgb


def pred_merge(img, pred, colors, mask_alpha=0.3, mode='RGB'):
    """Merge mask to pic."""
    img = np.array(img).astype(np.float32)
    idx = np.nonzero(pred)

    mask_color = color_encode(pred, colors, mode=mode)
    img[idx[0], idx[1], :] *= 1.0 - mask_alpha
    img[idx[0], idx[1], :] += mask_alpha * mask_color[idx[0], idx[1], :]

    return img.astype(np.uint8)


def unlock_movie(path):
    """ 将视频转换成图片
    path: 视频路径 """
    cap = cv2.VideoCapture(path)
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    suc, frame = cap.read()
    videoWriter = cv2.VideoWriter(out_path, fourcc, fps, (1280, 720))
    videoWriter.write(frame)
    while suc:
        frame_count += 1
        suc, frame = cap.read()
        frame = frame[:, :, ::-1]
        seg_mask = semseg_inference(frame)
        # mask = color_encode(mask, colors, mode='BGR')
        mask = pred_merge(frame, seg_mask, colors, 0.3, mode='RGB')
        mask = mask[:, :, ::-1]

        videoWriter.write(mask)
        if frame_count % 500 == 0:
            cv2.imwrite('{}/{}.png'.format(img_path, str(frame_count).zfill(5)), mask)
            print('decode num -- {}'.format(frame_count))
    videoWriter.release()

    cap.release()
    print('unlock movie: ', frame_count)


def make_video():
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # opencv3.0
    videoWriter = cv2.VideoWriter(out_path, fourcc, 30, (1280, 720))

    files = os.listdir(img_path)
    files.sort(key=lambda x: int(x.split('.')[0]))

    for i, path in enumerate(files):
        print(i)
        frame = cv2.imread(os.path.join(img_path, path))
        videoWriter.write(frame)
        if i == 500:
            break

    videoWriter.release()


if __name__ == '__main__':
    unlock_movie(video_path)
    # make_video()