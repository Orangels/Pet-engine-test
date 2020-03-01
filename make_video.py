import os
import cv2

import sys
import time
# sys.path.append('/home/user/workspace/priv-0220/Pet-engine')
sys.path.append('/home/user/workspace/Pet-engine-master/Pet-engine')
# sys.path.append('/home/user/workspace/priv-0220/Pet-dev')
from modules import pet_engine

confidence = 0.85

mode_type_car = 'car'
mode_type_fog = 'fog'
mode_type = [mode_type_car, mode_type_fog]
mode_index = 1

# video_path = '/home/user/Program/ls/video_test/20200226_125756_Trim.mp4'
video_path = '/home/user/Program/ls/video_test/20200226_125756_Trim_{}_test.mp4'.format(mode_type[abs(mode_index-1)])
# out_path = '/home/user/Program/ls/video_test/20200226_125756_Trim_{}_test.mp4'.format(mode_type[abs(mode_index)])
out_path = '/home/user/Program/ls/video_test/20200226_125756_Trim_test_merge.mp4'
img_path = '/home/user/workspace/priv-0220/privision_test/video_imgs/det_imgs'
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')  # opencv3.0
# fourcc = cv2.VideoWriter_fourcc('X','V','I','D')  # opencv3.0
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # 保存 mp4

print(pet_engine.MODULES.keys())
module = pet_engine.MODULES['SSDDet']
det = module(cfg_file='/home/user/workspace/priv-0220/Vas/yaml/ssd_VGG16_512x512_1x_vehicle/ssd_VGG16_512x512_1x_vehicle_{}.yaml'.format(mode_type[abs(mode_index)]),
             cfg_list=['VIS.VIS_TH', confidence,
                       'VIS.SHOW_BOX.COLOR_SCHEME',
                       None]
             )


def mat_inter(box1, box2):
    # 判断两个矩形是否相交
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False


def compare_boxex(box1, box2):
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    area1 = (x02 - x01) * (y02 - y01)
    area2 = (x12 - x11) * (y12 - y11)
    if area1 >= area2:
        return box1
    else:
        return box2


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

        output = det(frame)
        result_arr = []
        for i, item in enumerate(output):
            if type(item) is not list:
                output[i] = item.tolist()
                # print(output[i])
                for count_i, count_item in enumerate(output[i]):
                    for coor_i, coor_item in enumerate(count_item):
                        if coor_i != 4 and coor_i != 5:
                            coor_item = int(coor_item)
                            output[i][count_i][coor_i] = coor_item
                            # output[i][count_i][coor_i] = int(output[i][count_i][coor_i])
                        elif coor_i == 4:
                            if coor_item > confidence:
                                coor_item = int(coor_item * 100)
                                output[i][count_i][coor_i] = coor_item
                                result_arr.append(dict(label=i, box=count_item[:4], conf=coor_item))
                            # output[i][count_i][coor_i] = int(output[i][count_i][coor_i]*100)
                    # img = cv2.rectangle(img, (count_item[0], count_item[1]), (count_item[2], count_item[3]), (0, 255, 0), 2)
        print(result_arr)
        print(frame_count)

        # 根据 20200226_125756 视频人为过滤, 不适用其他视频
        if len(result_arr) == 2 and mat_inter(result_arr[0]['box'], result_arr[1]['box']):
            boxes = compare_boxex(result_arr[0]['box'], result_arr[1]['box'])
            print(boxes)
            frame = cv2.rectangle(frame, (boxes[0], boxes[1]), (boxes[2], boxes[3]), (214, 255, 26), 2)
            videoWriter.write(frame)
            if frame_count % 500 == 0:
                cv2.imwrite('{}/{}.png'.format(img_path, str(frame_count).zfill(5)), frame)
                print('decode num -- {}'.format(frame_count))
            continue

        for item in result_arr:
            coor = item['box']
            frame = cv2.rectangle(frame, (coor[0], coor[1]), (coor[2], coor[3]), (0, 0, 255), 2)
        # cv2.imwrite('{}/{}.png'.format(img_path, str(frame_count).zfill(5)), frame)
        videoWriter.write(frame)
        if frame_count % 500 == 0:
            cv2.imwrite('{}/{}.png'.format(img_path, str(frame_count).zfill(5)), frame)
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