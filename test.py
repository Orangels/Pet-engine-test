import sys
import cv2
import time
# sys.path.append('/home/user/workspace/priv-0220/Pet-engine')
sys.path.append('/home/user/workspace/Pet-engine-master/Pet-engine')
# sys.path.append('/home/user/workspace/priv-0220/Pet-dev')
from modules import pet_engine

confidence = 0.6



def compare_boxex(box1, box2):
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2
    area1 = (x02 - x01) * (y02 - y01)
    area2 = (x12 - x11) * (y12 - y11)
    if area1 >= area2:
        return box1
    else:
        return box2

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


def solve_coincide(box1, box2):
    # box=(xA,yA,xB,yB)
    # 计算两个矩形框的重合度
    if mat_inter(box1, box2) == True:
        x01, y01, x02, y02 = box1
        x11, y11, x12, y12 = box2
        col = min(x02, x12) - max(x01, x11)
        row = min(y02, y12) - max(y01, y11)
        intersection = col * row
        area1 = (x02 - x01) * (y02 - y01)
        area2 = (x12 - x11) * (y12 - y11)
        coincide = intersection / (area1 + area2 - intersection)
        return coincide
    else:
        return False
    

if __name__ == '__main__':
    print(pet_engine.MODULES.keys())
    module = pet_engine.MODULES['SSDDet']
    det = module(cfg_file='/home/user/workspace/priv-0220/Vas/yaml/ssd_VGG16_512x512_1x_vehicle/ssd_VGG16_512x512_1x_vehicle_fog.yaml',
                 cfg_list=['VIS.VIS_TH', confidence,
                           'VIS.SHOW_BOX.COLOR_SCHEME',
                           None]
                 )
    # det = module(
    #     cfg_file='/home/user/workspace/priv-0220/Vas/yaml/ssd_VGG16_300x300_1x/ssd_VGG16_300x300_car_1x.yaml',
    #     cfg_list=['VIS.VIS_TH', confidence,
    #               'VIS.SHOW_BOX.COLOR_SCHEME',
    #               None]
    #     )
    # car
    # img = cv2.imread('/home/user/workspace/priv-0220/Vas/test/ht_test.jpg')
    img = cv2.imread('03000.png')
    # img = cv2.imread('/home/user/workspace/priv-0220/privision_test/video_imgs/00500.png')

    # fog
    # img = cv2.imread('/home/user/workspace/priv-0220/Priv_fog/images/P000005.png')

    # 4 * n * 5  类别, 个数, 坐标+置信度
    output = det(img)
    time_start = time.time()
    print(output)
    print(len(output))
    result_arr = []
    for i, item in enumerate(output):
        if type(item) is not list:
            output[i] = item.tolist()
            print(output[i])
            for count_i, count_item in enumerate(output[i]):
                for coor_i, coor_item in enumerate(count_item):
                    if coor_i != 4 and coor_i != 5:
                        coor_item = int(coor_item)
                        output[i][count_i][coor_i] = coor_item
                        # output[i][count_i][coor_i] = int(output[i][count_i][coor_i])
                    elif coor_i == 4:
                        if coor_item > confidence:
                            coor_item = int(coor_item*100)
                            output[i][count_i][coor_i] = coor_item
                            result_arr.append(dict(label=i, box=count_item[:4], conf=coor_item))
                        # output[i][count_i][coor_i] = int(output[i][count_i][coor_i]*100)
                # img = cv2.rectangle(img, (count_item[0], count_item[1]), (count_item[2], count_item[3]), (0, 255, 0), 2)
    print(result_arr)
    print('cost time {}'.format(time.time()-time_start))
    
    # 根据 20200226_125756 视频人为过滤, 不适用其他视频
    if len(result_arr) == 2 and mat_inter(result_arr[0]['box'], result_arr[1]['box']):
        print('merge')
        boxex = compare_boxex(result_arr[0]['box'], result_arr[1]['box'])
        print(boxex)
        img = cv2.rectangle(img, (boxex[0], boxex[1]), (boxex[2], boxex[3]), (214, 255, 26), 2)

    for item in result_arr:
        coor = item['box']
        img = cv2.rectangle(img, (coor[0], coor[1]), (coor[2], coor[3]), (0, 255, 0), 2)
    cv2.imwrite('222.png', img)