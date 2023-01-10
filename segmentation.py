#基于肤色的手势图像分割算法
import numpy as np
import cv2
import imutils

hand_hsv_lowrange = np.array([0, 48, 50])
hand_hsv_highrange = np.array([20, 255, 255])

def kai_process(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    res = cv2.erode(image, kernel)
    res = cv2.dilate(res, kernel)
    return res

def bi_process(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    res = cv2.dilate(image, kernel)
    res = cv2.erode(res, kernel)
    return res

def hand_YCrCb_Otsu_segmentation(image_bgr):
    fgbg = cv2.createBackgroundSubtractorMOG2()  # 利用BackgroundSubtractorMOG2算法消除背景
    fgmask = fgbg.apply(image_bgr)
    image_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=fgmask)
    image_YCrCb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCR_CB)#将bgr图像转换到Ycrcb空间
    (Y, Cr, Cb) = cv2.split(image_YCrCb)#分离Y，Cr，Cb三个分量
    #cr_blur = cv2.GaussianBlur(Cr, (5, 5), 0)#对Cr分量做高斯模糊去除噪声
    cr_blur = cv2.bilateralFilter(Cr, 9, 75, 75)
    _, skin = cv2.threshold(cr_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)#根据Cr分量进行Otus自适应阈值处理二值化
    res = cv2.bitwise_and(image_bgr, image_bgr, mask=skin)#根据Cr分量的阈值分割原图像
    res = bi_process(res, 7)  #闭运算：1、膨胀 2、腐蚀
    return res

def hand_YCrCb_selfsetting_segmentation(image_bgr, thresshold):
    image_YCrCb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCR_CB)  # 将bgr图像转换到Ycrcb空间
    (Y, Cr, Cb) = cv2.split(image_YCrCb)  # 分离Y，Cr，Cb三个分量
    cr_blur = cv2.GaussianBlur(Cr, (9, 9), 0)  # 对Cr分量做高斯模糊去除噪声
    # cr_blur = cv2.bilateralFilter(Cr, 9, 75, 75)
    _, skin = cv2.threshold(cr_blur, thresshold, 255, cv2.THRESH_BINARY)
    res = cv2.bitwise_and(image_bgr, image_bgr, mask=skin)  # 根据Cr分量的阈值分割原图像
    res = kai_process(res, 5)  # 采用开运算=先腐蚀，再膨胀
    return res

def hand_YCrCb_ellipse_segmentation(image_rgb):
    skinCrCbHist = np.zeros((256, 256), dtype=np.uint8)
    cv2.ellipse(skinCrCbHist, (113, 155), (23, 25), 43, 0, 360, (255, 255, 255), -1)#绘制椭圆弧线
    image_YCrCb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2YCR_CB)
    (Y, Cr, Cb) = cv2.split(image_YCrCb)
    skin = np.zeros(Cr.shape, dtype=np.uint8)
    x, y = Cr.shape
    for i in range(0,x):
        for j in range(0,y):
            if skinCrCbHist [Cr[i][j], Cb[i][j]] > 0:
                skin[i][j] =255
    res = cv2.bitwise_and(image_rgb, image_rgb, mask =skin)
    return res

def hand_HSV_segmentation(image_bgr):
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(image_hsv, hand_hsv_lowrange, hand_hsv_highrange)
    skin = kai_process(skin, 3)
    skin = cv2.GaussianBlur(skin, (15,15), 1)

    res = cv2.bitwise_and(image_bgr, image_bgr, mask = skin)
    return res

def contour_Laplacian_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(image_gray, cv2.CV_16S, ksize=3)
    dst_abs = cv2.convertScaleAbs(dst)
    binary_image = cv2.Canny(dst_abs, 50, 200)
    profile = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = profile[0]
    contour = sorted(contour, key = cv2.contourArea, reverse = True)
    #contour_hand = contour[0]
    white_background = np.ones(dst.shape, np.uint8)
    res = cv2.drawContours(white_background, contour, -1, (255, 255, 255), 3)
    return res



#调用笔记本摄像头进行视频demo展示图像分割效果
def video_demo():
    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print("Cannot open camera")
        exit()
    while(True):
        ref, frame = capture.read()#读取笔记本摄像头

        target_window = cv2.resize(frame, (400, 400), interpolation=cv2.INTER_CUBIC)#调节显示窗口大小
        cv2.rectangle(target_window, (90, 110), (300, 350), (0, 255, 0))#选取需要处理的图像部分
        image_rgb = target_window[110:350, 90:300]

        hand_RGB = image_rgb
        hand_segmetation = hand_YCrCb_Otsu_segmentation(hand_RGB)#通过YCrCb的Cr分量+Otsu阈值处理，分割手势
        hand_contour = contour_Laplacian_detection(hand_segmetation)
        image_comparison = np.hstack([hand_RGB, hand_segmetation])

        cv2.namedWindow("video_demo", 0)
        cv2.imshow("video_demo", image_comparison)
        cv2.imshow("hand_contour", hand_contour)
        cv2.imshow("camera", target_window)

        c = cv2.waitKey(30) & 0xff#等待30s或者按esc退出demo
        if c == 27:
            capture.release()
            break

# video_demo()
# cv2.destroyAllWindows()











