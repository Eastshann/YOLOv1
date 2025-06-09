import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from Backbone.Backbone import *
from Block.CBL import *
from Head.Head import *

class Yolo(nn.Module):
    """
    Yolo网络由backbone和head构成，backbone输出7x7x1024，head输出7x7x30
    """
    def __init__(self, num_classes=20):
        super(Yolo, self).__init__()
        self.backbone = Backbone()
        self.head = Head(num_classes)
        
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 计算用于初始化权重的方差因子，使用He初始化方法（适合ReLU）
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                 # 使用正态分布初始化卷积核的权重，均值为0，标准差为 sqrt(2 / n)
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    # 如果卷积层有偏置项，则将偏置初始化为0
                    m.bias.data.zero_()
                    
            elif isinstance(m, nn.BatchNorm2d):
                # 将BatchNorm的权重初始化为1（这样初始时不改变激活分布）
                m.weight.data.fill_(1)
                # 将偏置初始化为0
                m.bias.data.zero_()
                
            elif isinstance(m, nn.Linear):
                # 使用正态分布初始化权重，均值为0，标准差为0.01（比卷积层小）
                m.weight.data.normal_(0, 0.01)
                # 将偏置初始化为0
                m.bias.data.zero_()

    def forward(self, x):
        x = self.backbone(x)
        # batch_size * channel * width * height
        x = x.permute(0, 2, 3, 1)
        x = torch.flatten(x, start_dim=1, end_dim=3)  # 平铺向量
        x = self.head(x)
        x = F.sigmoid(x) # 归一化到0-1
        x = x.view(-1,7,7,30) # 重塑成bs,7,7,30张量
        return x
