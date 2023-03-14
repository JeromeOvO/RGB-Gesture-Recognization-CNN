#import keras
import segmentation as seg
import cv2
import numpy as np
import picture_capture as pc
import os
import torchvision.transforms as transforms
import h5py



cap = cv2.VideoCapture(0)
def data_capture():
    start = 0
    time_count = 0
    sum = 1437

    while True:
        ret, img = cap.read()
        img_capture = pc.find_yourhands(img, 100, 200, 200, 200)
        img_seg = seg.hand_YCrCb_Otsu_segmentation(img_capture)
        cv2.imshow("train_img", img_seg)
        cv2.imshow("camera", img)
        if start == 1 and time_count % 5 == 0:
            cv2.imwrite(r'D:\Undergraduate\Mycode\Dataset\train_data\train_'+str(sum)+'.jpg', img_seg)
            print('write'+str(sum)+'completed')
            sum += 1
        time_count += 1
        key = cv2.waitKey(10)
        if key == ord('s'):
            start = 1
            print('Start capture.................')
        if key == 27: break #press esc
    cv2.destroyAllWindows()
    print('Data-capture Closed.............')


def read_dataset(data_path):
    zero = []
    label_zero = []
    one = []
    label_one = []
    two = []
    label_two = []
    three = []
    label_three = []
    four = []
    label_four = []
    five = []
    label_five = []

    print('Loading Database ..................')

    for file in os.listdir(data_path + '/00'):
        zero.append(data_path + '/00' + '/' + file)
        label_zero.append(0)
    for file in os.listdir(data_path + '/01'):
        one.append(data_path + '/01' + '/' + file)
        label_one.append(1)
    for file in os.listdir(data_path + '/02'):
        two.append(data_path + '/02' + '/' + file)
        label_two.append(2)
    for file in os.listdir(data_path + '/03'):
        three.append(data_path + '/03' + '/' + file)
        label_three.append(3)
    for file in os.listdir(data_path + '/04'):
        four.append(data_path + '/04' + '/' + file)
        label_four.append(4)
    for file in os.listdir(data_path + '/05'):
        five.append(data_path + '/05' + '/' + file)
        label_five.append(5)

    print('Creating dataset ..................')
    image_list = np.hstack((zero, one, two, three, four, five))
    label_list = np.hstack((label_zero, label_one, label_two, label_three, label_four, label_five))

    # temp = np.array([label_list, image_list])
    # temp = temp.transpose()
    #
    # np.random.shuffle(temp)

    counter = 0
    x = []
    for dirs in image_list:
        counter = counter + 1
        image = cv2.imread(dirs)
        image = image.astype(np.float32)
        #image /= 255.0
        image = cv2.resize(image, (100, 100))
        print("Processing No. %d image" % counter)
        mat = np.asarray(image)
        x.append(mat)

    aa = np.array(x)
    # num, _, _ = aa.shape
    # aa.reshape((num, 100, 100, 1))

    image_data = aa
    label_data = label_list

    # file = h5py.File(r"D:\Undergraduate\Mycode\Dataset\train_data\h5_data//data_h5.h5","w")
    # file.create_dataset('X', data=aa)
    # file.create_dataset('Y', data=np.array(label_list))
    # file.close()

    print('Success.........................')
    return image_data, label_data

def image_roate(image, image_outpath):
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    dst = cv2.warpAffine(image, M, (cols, rows))
    cv2.imwrite(image_outpath+r"\00.jpg", dst)

# def data_augment(database_bath):
#
#     for file in os.listdir(database_bath + '/00'):
#         zero.append(data_path + '/00' + '/' + file)
#         label_zero.append(0)
#     for file in os.listdir(database_bath + '/01'):
#         one.append(data_path + '/01' + '/' + file)
#         label_one.append(1)
#     for file in os.listdir(database_bath + '/02'):
#         two.append(data_path + '/02' + '/' + file)
#         label_two.append(2)
#     for file in os.listdir(database_bath + '/03'):
#         three.append(data_path + '/03' + '/' + file)
#         label_three.append(3)
#     for file in os.listdir(database_bath + '/04'):
#         four.append(data_path + '/04' + '/' + file)
#         label_four.append(4)
#     for file in os.listdir(database_bath + '/05'):
#         five.append(data_path + '/05' + '/' + file)
#         label_five.append(5)




if __name__ == "__main__":
    #image = cv2.imread(r"D:\Undergraduate\Mycode\Dataset\original\0.jpg")
    data_capture()#通过摄像头生成数据图片，自动存储，按s开始捕获，按esc退出
    #image_roate(image, r"D:\Undergraduate\Mycode\Dataset\original")
#     image_data, label_data = read_dataset(r'D:\Undergraduate\Mycode\Dataset\train_data')
#     print(image_data.shape, label_data.shape)
#     print(image_data.dtype, label_data.dtype)
#     print(label_data)
#     img = cv2.imread(r'D:\Undergraduate\Mycode\Dataset\train_data\00\train_106.jpg')
#     img_to_tensor = transforms.ToTensor()
#     img = img_to_tensor(img)
#     print(img.shape)