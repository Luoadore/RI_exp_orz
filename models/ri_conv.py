# coding: utf-8

from torch import nn

class share_conv_3X3(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(share_conv_3X3, self).__init__()
        if type(stride)==type(0):
            stride = (stride,stride)
        self.stride=stride
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=1, stride=stride, bias=bias)
        self.conv2 = nn.Conv2d(in_channel,out_channel,kernel_size=1, stride=stride, bias=bias)
        self.conv3 = nn.Conv2d(in_channel,out_channel,kernel_size=1, stride=stride, bias=bias)
    def forward(self, x):
        h = x.size(2)-(x.size(2)-3) % self.stride[0]
        w = x.size(3)-(x.size(3)-3) % self.stride[1]
        x1 = self.conv1(x[:,:,0:h-2,0:w-2]) + self.conv1(x[:,:,2:h,2:w]) + self.conv1(x[:,:,0:h-2,2:w]) + self.conv1(x[:,:,2:h,0:w-2])
        x2 = self.conv2(x[:,:,0:h-2,1:w-1]) + self.conv2(x[:,:,2:h,1:w-1])+ self.conv2(x[:,:,1:h-1,0:w-2]) + self.conv2(x[:,:,1:h-1,2:w])
        x3 = self.conv3(x[:,:,1:h-1,1:w-1])
        x = (x1+x2+x3)/9
        return x