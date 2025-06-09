import torch.nn as nn

class Head(nn.Module):
    """
    检测头由两个全连接层构成，第一层 7x7x1024->4096, 第二层4096->7x7*30
    """
    def __init__(self, num_classes=20):
        super(Head, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(7*7*1024, 2048, bias=True), # 这里偏置不能少
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, (num_classes+2*5)*7*7, bias=True),  # 修改了4096->2048
        )
        
    def forward(self, x):
        return self.classifier(x)
