import cv2
import numpy as np

MIN_DESCRIPTOR = 128#设置最小数量项傅里叶系数

def fourierDesciptor(image):#计算傅里叶描述子
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    dst = cv2.Laplacian(image_gray, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    profile = cv2.findContours(Laplacian, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = profile[1]
    contour = sorted(contour, key=cv2.contourArea, reverse=True)
    contour_array = contour[0][:, 0, :]#保留区域面积最大的轮廓点的坐标,去噪声
    black_background = np.ones(dst.shape, np.uint8)#创建黑色幕布
    res = cv2.drawContours(black_background, contour[0], -1, (255, 255, 255), 1)
    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]#横坐标为实数部分
    contour_complex.imag = contour_array[:, 1]#纵坐标为虚数部分
    fourier_res = np.fft.fft(contour_complex)#进行傅里叶变换
    descriptor_used = truncate_descriptor(fourier_res)#截短傅里叶描述子

    return res, descriptor_used

def truncate_descriptor(fourier_res):#截短傅里叶描述子函数
    descriptor_used = np.fft.fftshift(fourier_res)
    center_index = int(len(descriptor_used)/2)
    low, high = center_index - int(MIN_DESCRIPTOR/2), center_index + int(MIN_DESCRIPTOR/2)
    descriptor_used = descriptor_used[low:high]
    descriptor_used = np.fft.ifftshift(descriptor_used)

    return descriptor_used

def reconstruct_fourier(img, descriptor_used):
    contour_reconstruction = np.fft.ifft(descriptor_used)#反傅里叶变换
    contour_reconstruction = np.array([contour_reconstruction.real, contour_reconstruction.imag])
    contour_reconstruction = np.transpose(contour_reconstruction)#转置
    contour_reconstruction = np.expand_dims(contour_reconstruction, axis=1)

    if contour_reconstruction.min() < 0 :
        contour_reconstruction -= contour_reconstruction.min()

    contour_reconstruction *= img.shape[0]/contour_reconstruction.max()
    contour_reconstruction = contour_reconstruction.astype(np.int32, copy=False)

    blackbackground = np.ones(img.shape, np.uint8)
    res = cv2.drawContours(blackbackground, contour_reconstruction, -1, (255, 255, 255), 1)
    return res



