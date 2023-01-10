import torch
import torch.nn as nn
import torch.nn.functional as F

def evaluate(model, test_loader, BATCH_SIZE):
    correct = 0
    for test_imgs, test_labels in test_loader:
        output = model(test_imgs.float())
        predicted = torch.max(output, 1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{: .3f}% ".format(float(correct)/(len(test_loader)*BATCH_SIZE)*100))



class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        #Convolution
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6,out_channels=16, kernel_size=5, stride=1, padding=0)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)
        #Full connect
        self.fc1 = torch.nn.Linear(16*23*23, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)

        x = x.view(x.size(0), -1)#x.view(变成几行， 变成几列)，若不确定，填入-1，自适应展开

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



