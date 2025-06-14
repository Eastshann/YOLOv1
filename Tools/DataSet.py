'''
txt描述文件 image_name.jpg x y w h c x y w h c 这样就是说一张图片中有两个目标
'''
import os
import os.path

import numpy as np

import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import cv2


class YoloDataset(data.Dataset):
    image_size = 448
    def __init__(self, root, list_file, train, transform=None):
        print('data init')
        self.root = root
        self.train = train
        self.transform = transform
        self.fnames = []
        self.boxes = []
        self.labels = []
        self.mean = (123, 117, 104)  # RGB

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            splited = line.strip().split()
            # 文件名
            self.fnames.append(splited[0])
            # bbox数
            num_boxes = (len(splited) - 1) // 5
            box = []
            label = []
            for i in range(num_boxes):
                x = float(splited[1 + 5 * i])
                y = float(splited[2 + 5 * i])
                x2 = float(splited[3 + 5 * i])
                y2 = float(splited[4 + 5 * i])
                c = splited[5 + 5 * i]
                box.append([x, y, x2, y2])
                label.append(int(c) + 1)
            
            # 转换为tensor
            self.boxes.append(torch.Tensor(box))
            self.labels.append(torch.LongTensor(label))
        
        # 图片总数
        self.num_samples = len(self.boxes)

    def __getitem__(self, idx):
        # 构建图片路径
        fname = self.fnames[idx]
        img_path = os.path.join(self.root, 'images', fname)
        # 读取图片
        img = cv2.imread(img_path)
        boxes = self.boxes[idx].clone()
        labels = self.labels[idx].clone()
        # 归一化坐标
        h, w, _ = img.shape
        boxes /= torch.Tensor([w, h, w, h]).expand_as(boxes)  # 归一化坐标
        img = self.BGR2RGB(img)  # because pytorch pretrained model use RGB
        img = self.subMean(img, self.mean)  # 减去均值
        img = cv2.resize(img, (self.image_size, self.image_size))
        # 编码
        target = self.encoder(boxes, labels)  # 7x7x30
        # 图像增强
        for t in self.transform:
            img = t(img)

        return img, target

    # 必要，数据迭代时需要len()
    def __len__(self):
        return self.num_samples

    def encoder(self, boxes, labels):
        '''
        boxes (tensor) [[x1,y1,x2,y2],[]] ex:tensor([[0.0500, 0.1024, 0.8380, 0.8163]])
        labels (tensor) [...]
        return 7x7x30
        '''
        grid_num = 7
        target = torch.zeros((grid_num, grid_num, 30)) # 初始化 7×7×30 的输出目标
        """
        [0:2]     → 第一个框的中心坐标 (x, y)
        [2:4]     → 第一个框的宽高     (w, h)
        [4]       → 第一个框的置信度 confidence

        [5:7]     → 第二个框的中心坐标 (x, y)
        [7:9]     → 第二个框的宽高     (w, h)
        [9]       → 第二个框的置信度 confidence

        [10:30]   → 20 维的 one-hot 类别信息
        """
        cell_size = 1. / grid_num # 每个 cell 的宽度/高度（归一化后）
        wh = boxes[:, 2:] - boxes[:, :2]  # 计算标注框wh，[:, 2:]为每一行的右下角坐标，[:, :2]为每一行的坐上角坐标，减 1 是为了将 ceil 后的坐标转换为 0-based 的索引
        cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2  # 标注框中心坐标
        for i in range(cxcy.size()[0]):
            cxcy_sample = cxcy[i]
            ij = (cxcy_sample / cell_size).ceil() - 1  # 得到该中心点在哪个 cell 里，【.ceil()】为向上取整
            
            target[int(ij[1]), int(ij[0]), 4] = 1   # 第一个 bbox 的 confidence = 1，注：这个是bbox的置信度
            target[int(ij[1]), int(ij[0]), 9] = 1   # 第二个 bbox 的 confidence = 1，注：这个是bbox的置信度
            target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1  # one-hot 类别，注：这个是类别的置信度
            
            xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
            delta_xy = (cxcy_sample - xy) / cell_size  # 相对于cell左上角偏移量，是目标中心点相对于网格左上角的偏移（归一化后）
            
            target[int(ij[1]), int(ij[0]), 2:4] = wh[i]     # 第一个框的中心点
            target[int(ij[1]), int(ij[0]), :2] = delta_xy   # 第一个框的 w,h
            target[int(ij[1]), int(ij[0]), 7:9] = wh[i]     # 第二个框的中心点
            target[int(ij[1]), int(ij[0]), 5:7] = delta_xy  # 第二个框的 w,h
            
        return target

    def BGR2RGB(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def subMean(self, bgr, mean):
        mean = np.array(mean, dtype=np.float32)
        bgr = bgr - mean
        return bgr


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    root = 'D:\codes\python\Yolov1\datasets'
    list_file = os.path.join(root, 'train.txt')
    train_dataset = YoloDataset(root, list_file, train=True, transform=[transforms.ToTensor()])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
    train_iter = iter(train_loader)
    for i in range(10):
        img, target = next(train_iter)
        print(img.shape)

