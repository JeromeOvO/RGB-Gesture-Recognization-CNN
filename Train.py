import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.utils.data
from MyModel import hand_rcnn, evaluate
from LeNet5_model import LeNet5, evaluate
from sklearn.model_selection import train_test_split
import h5py
import Dataset.dataset_build as db
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

Trans_img = transforms.ToTensor()
X_data, Y_data = db.read_dataset(r'D:\Undergraduate\Mycode\Dataset\train_data')
# print(X_data.shape)
# print(Y_data.shape)
x_train, x_test, y_train, y_test = train_test_split(X_data, Y_data, train_size=0.7, test_size=0.3, random_state=42)

BATCH_SIZE = 16

#turn data type
torch_x_train = torch.from_numpy(x_train)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
torch_x_test = torch.from_numpy(x_test)
torch_y_test = torch.from_numpy(y_test).type(torch.LongTensor)
# print(torch_x_train.size())
# print(torch_y_train.size())

#for cnn
torch_x_train = torch_x_train.view(-1, 3, 100, 100)
torch_x_test = torch_x_test.view(-1, 3, 100, 100)
# print(torch_x_train.size())

#build dataset
train_data = torch.utils.data.TensorDataset(torch_x_train, torch_y_train)
test_data = torch.utils.data.TensorDataset(torch_x_test, torch_y_test)
#build dataloader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

def Train_Model(model, train_loader, Epoches):
    optimizer = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    EPOCHES = Epoches

    for epoch in range(EPOCHES):
        correct = 0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            #print(x_batch.shape)
            output = model(x_batch)
            #print(output.shape, y_batch.shape)
            loss = loss_func(output, y_batch)
            loss.backward()
            optimizer.step()
            predict = torch.max(output.data, 1)[1]
            correct += (predict == y_batch).sum()

            if batch_idx % 10 == 0:
                    print('Epoch: {} [{}/{}({:.0f}%)]\t Loss: {:.6f}\t Accuracy: {:.3f}%'.format(epoch,
                                                                                                 batch_idx * BATCH_SIZE,
                                                                                                 (len(train_data)),
                                                                                                 batch_idx * BATCH_SIZE / (
                                                                                                     len(train_data)) * 100,
                                                                                                 loss,
                                                                                                 float(correct) / (
                                                                                                         len(train_loader) * BATCH_SIZE) * 100))
if __name__=="__main__":
    mymodel = hand_rcnn()
    # mymodel = LeNet5()
    print(mymodel)
    Train_Model(mymodel, train_loader, 20)
    print("Train Success .......................................")
    evaluate(mymodel, test_loader, BATCH_SIZE)
    torch.save(mymodel, "./LeNet-5.pkl")
    print("Save Net Success .....................................")


