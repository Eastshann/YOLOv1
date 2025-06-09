import torch.nn as nn

class CBL(nn.Module):
    """
    Conv-BN-LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride = 1, padding = 0):
        super(CBL, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False) # 使用了BN，不需要偏置
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(inplace=True) # 原地操作数据，减少内存开销，提升速度
        
    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        return x
    
    