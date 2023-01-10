import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T

def evaluate(model, test_loader, BATCH_SIZE):
    correct = 0
    for test_imgs, test_labels in test_loader:
        output = model(test_imgs.float())
        predicted = torch.max(output, 1)[1]
        correct += (predicted == test_labels).sum()
    print("Test accuracy:{: .3f}% ".format(float(correct)/(len(test_loader)*BATCH_SIZE)*100))

class hand_rcnn(nn.Module):
    def __init__(self):
        super(hand_rcnn, self).__init__()
        #conv connect
        #w = (w - f + 2* padding)/stride + 1
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.max_pool1 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64,kernel_size=3, stride=1, padding=1)
        self.max_pool2 = torch.nn.MaxPool2d(kernel_size=2)
        #full connect
        self.fc1 = torch.nn.Linear(64*25*25, 120)
        self.fc2 = torch.nn.Linear(120, 80)
        self.fc3 = torch.nn.Linear(80, 6)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        #x = F.relu(self.conv3(x))
        #x = self.max_pool3(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # x = F.relu(self.fc2(x))
        #x = self.soft_max(x)

        return x
