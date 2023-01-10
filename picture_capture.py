import cv2
import numpy as np



def find_yourhands(frame, x0, y0, width, height):
    cv2.rectangle(frame, (x0, y0), (x0 + width, y0 + height), (0, 255, 0))  # 画出截取的手势框图
    res = frame[y0:y0 + height, x0:x0 + width]  # 获取手势框图
    return res

def absdiff_process(image_1, image_2, sThre):
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2GRAY)  #灰度化
    gray_image_1 = cv2.GaussianBlur(gray_image_1, (3, 3), 0)  #高斯滤波
    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2GRAY)
    gray_image_2 = cv2.GaussianBlur(gray_image_2, (3, 3), 0)
    d_frame = cv2.absdiff(gray_image_1, gray_image_2)
    ret, d_frame = cv2.threshold(d_frame, sThre, 255, cv2.THRESH_BINARY)
    d_frame = cv2.dilate(d_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4)), iterations=2)
    d_contours = cv2.findContours(d_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return d_contours

def absdiff_process2(image_1, image_2, sThre):
    gray_image_1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2YCR_CB)  #灰度化
    (Y1, Cr1, Cb1) = cv2.split(gray_image_1)  # 分离Y，Cr，Cb三个分量
    Cr1 = cv2.GaussianBlur(Cr1, (3, 3), 0)  #高斯滤波
    gray_image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2YCR_CB)
    (Y2, Cr2, Cb2) = cv2.split(gray_image_2)  # 分离Y，Cr，Cb三个分量
    Cr2 = cv2.GaussianBlur(Cr2, (3, 3), 0)

    d_frame = cv2.absdiff(Cr1, Cr2)
    ret, d_frame = cv2.threshold(d_frame, sThre, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    d_frame = cv2.dilate(d_frame, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4)), iterations=2)
    d_contours = cv2.findContours(d_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return d_contours




capture = cv2.VideoCapture(0)
sThre = 25 #sThre表示像素阈值

# while(True):
#     ret, frame = capture.read()
#     cv2.waitKey(20)
#     ret_2, frame_2 = capture.read()
#     d_frame = absdiff_demo(frame, frame_2, sThre)
#     cv2.imshow("1", d_frame)

