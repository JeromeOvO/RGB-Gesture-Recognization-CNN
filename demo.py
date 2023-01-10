import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from MyModel import hand_rcnn, evaluate
from sklearn.model_selection import train_test_split
import picture_capture as pc
import cv2
import segmentation as seg

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    MyNet = torch.load('./MyNet.pkl')
    while True:
        ret, img = cap.read()
        img_capture = pc.find_yourhands(img, 50, 50, 200, 200)
        img_seg = seg.hand_YCrCb_selfsetting_segmentation(img_capture, 140)
        image = img_seg.astype(np.float32)
        image = cv2.resize(image, (100, 100))
        image = np.array(image)
        image = torch.from_numpy(image)
        image = image.view(-1, 3, 100, 100)
        cv2.imshow("train_img", img_seg)
        cv2.imshow("camera", img)
        output = MyNet(image)
        predict = torch.max(output, 1)[1].data.numpy()
        print(output)
        print(predict)
        key = cv2.waitKey(10)
        if key == ord('s'):
            start = 1
            print('Start capture.................')
        if key == 27: break  # press esc
    cv2.destroyAllWindows()
    print('Demo Closed.............')




