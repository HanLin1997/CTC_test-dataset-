import torch
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 120, kernel_size=3)
        self.batch_norm1 = nn.BatchNorm2d(120)
        self.conv2 = nn.Conv2d(120, 100, kernel_size=3)
        self.batch_norm2 = nn.BatchNorm2d(100)
        self.conv3 = nn.Conv2d(100, 85, kernel_size=3)
        self.batch_norm3 = nn.BatchNorm2d(85)
        self.conv4 = nn.Conv2d(85, 66, kernel_size=3)
        self.batch_norm4 = nn.BatchNorm2d(66)


    def forward(self, x):
        #x = x.view(8, 1, 28, 28)
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  #(24, 259)
        x = self.batch_norm1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  #(11, 128)
        x = self.batch_norm2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)  #(4, 63)
        x = self.batch_norm3(x)
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)  #(1, 30)
        x = self.batch_norm4(x)
        return F.log_softmax(x, dim=1)
    
if __name__ == '__main__':
    x = torch.randn(2, 1, 50, 120)
    net = Net()
    y = net(x)
    #y = y.view(30, 2, 66)
    print(y.shape)