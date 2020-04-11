import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np 
import cv2
nConv=3 #you can choose more or less conv
class MyNet(nn.Module):
    def __init__(self,input_dim):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, 100, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(100)
        self.conv2 = []
        self.bn2 = []
        for i in range(nConv-1):
            self.conv2.append(nn.Conv2d(100, 100, kernel_size = 3, stride = 1, padding = 1 ) )
            self.bn2.append( nn.BatchNorm2d(100) )
        self.conv3 = nn.Conv2d(100, 100, kernel_size = 1, stride = 1, padding = 0 )
        self.bn3 = nn.BatchNorm2d(100)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

## Compute classes and centroids
def centroid_histogram(clt):
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins = numLabels)
    hist = hist.astype("float")
    hist /= hist.sum()
    return hist

## Plot average cluster colors
def plot_colors(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    startX = 0
    percentage = []
    for (percent, color) in zip(hist, centroids):
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
            color.astype("uint8").tolist(), -1)
        startX = endX
        percentage.append(percent)
    return bar, percentage
