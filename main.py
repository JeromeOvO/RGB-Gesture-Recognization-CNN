import cv2
import numpy as np
import segmentation as seg
import picture_capture as pic
import fourier as fd
import torch
import logging
import time
import sys
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie import Crazyflie
from cflib.positioning.motion_commander import MotionCommander
from fly import control_UOV
import multiprocessing
import threading
# URI = 'radio://0/100/2M/E7E7E7E701'
# # Only output errors from the logging framework
# logging.basicConfig(level=logging.ERROR)


WINDOWS_WIDTH_MAX = 640
WINDOWS_HEIGHT_MAX = 480
font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体
size = 0.5  # 设置大小

width, height = 180, 180  # 设置拍摄窗口大小
x0, y0 = 60, 140  # 设置选取位置

hand_position_x = 0
hand_position_y = 0
hand_position_w = 0
hand_position_h = 0
cap = cv2.VideoCapture(0)  # 开摄像头

filter_num = 10
background = None
count = [1, 1, 1, 1, 1, 1]

if __name__ == "__main__":
    start = 0
    print("Program Running!!!!!!!!!")
    MyNet = torch.load('./MyNet.pkl')
    print("Loading Net Success!!!")

    UOV = control_UOV()
    UOV.stop()
    print("Control UOV!!!!!")

    while (True):
        ret, frame = cap.read()  # 读取摄像头的内容

        text = 'Hand: '

        # 创建中间关键帧
        cv2.waitKey(30)
        _, frame2 = cap.read()

        # 采用目标跟踪算法自动寻找目标区域
        key = cv2.waitKey(1) & 0xff
        # 按'q'健退出循环
        if key == 27:
            break

        # 采用帧间差分算法寻找目标区域
        d_contours = pic.absdiff_process(frame2, frame, 50)
        for c in d_contours[1]:
            if cv2.contourArea(c) < 3500:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
                continue
            hand_position_x, hand_position_y, hand_position_w, hand_position_h = cv2.boundingRect(c)

        #找到手势位置
        image_rgb = pic.find_yourhands(frame, hand_position_x, hand_position_y, 160, 160)

        # 图像预处理
        image_segmentation = seg.hand_YCrCb_Otsu_segmentation(image_rgb)#图像分割
        #image_segmentation = seg.hand_YCrCb_selfsetting_segmentation(image_rgb, 140)  # 图像分割
        image_comparison = np.hstack([image_rgb, image_segmentation])

        #processing your hand image
        image = image_segmentation.astype(np.float32)
        image = cv2.resize(image, (100, 100))
        image = np.array(image)
        image = torch.from_numpy(image)
        image = image.view(-1, 3, 100, 100)

        if key == ord('s'):
            start = 1
            print('Start Controlling.................')
        if start == 1:
            #predict your hand
            output = MyNet(image)
            predict = torch.max(output, 1)[1].data.numpy()

            count[predict[0]] += 1

            print("Your hand is:", predict)
            text += str(predict)
            cv2.putText(frame, text, (hand_position_x, hand_position_y), cv2.FONT_ITALIC, 0.75, (0, 255, 0), 3)

            if count[0] % filter_num == 0:
                count[0] = 1
                print("######################################################################")
                UOV.turnning_four_paddle()
            elif count[1] % filter_num == 0:
                count[1] = 1
                print("######################################################################")
                UOV.move_process_1()
            elif count[2] % filter_num == 0:
                count[2] = 1
            elif count[3] % filter_num == 0:
                count[3] = 1
            elif count[4] % filter_num == 0:
                count[4] = 1
            elif count[5] % filter_num == 0:
                count[5] = 1

        if key == ord('z'):
            print('Stop Controlling.................')
            start = 0

        cv2.namedWindow("comparison", 0)
        cv2.imshow("frame", frame)  # 播放摄像头的内容
        cv2.imshow("comparison", image_comparison)


        # # 拉普拉斯算子提取手势轮廓
        # image_contour = seg.contour_Laplacian_detection(image_segmentation)
        # cv2.imshow("hand_contour_laplacian", image_contour)

        # 傅里叶描述子进行特征提取
        # res, descriptor_used = fd.fourierDesciptor(image_segmentation)
        # fourier_contour = fd.reconstruct_fourier(res, descriptor_used)
        # cv2.imshow("fourier_contour", fourier_contour)

    cap.release()
    cv2.destroyAllWindows()  # 关闭所有窗口
    print("Program Closed..............")