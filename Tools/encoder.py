import torch

def encoder(self, boxes, labels):
    '''
    boxes (tensor) [[x1,y1,x2,y2],[]] ex:tensor([[0.0500, 0.1024, 0.8380, 0.8163]])
    labels (tensor) [...]
    return 7x7x30
    '''
    grid_num = 7
    target = torch.zeros((grid_num, grid_num, 30))
    cell_size = 1. / grid_num
    wh = boxes[:, 2:] - boxes[:, :2]  # 计算标注框wh
    cxcy = (boxes[:, 2:] + boxes[:, :2]) / 2  # 标注框中心坐标
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        ij = (cxcy_sample / cell_size).ceil() - 1  #
        target[int(ij[1]), int(ij[0]), 4] = 1
        target[int(ij[1]), int(ij[0]), 9] = 1
        target[int(ij[1]), int(ij[0]), int(labels[i]) + 9] = 1
        xy = ij * cell_size  # 匹配到的网格的左上角相对坐标
        delta_xy = (cxcy_sample - xy) / cell_size  # 相对于cell左上角偏移量
        target[int(ij[1]), int(ij[0]), 2:4] = wh[i]
        target[int(ij[1]), int(ij[0]), :2] = delta_xy
        target[int(ij[1]), int(ij[0]), 7:9] = wh[i]
        target[int(ij[1]), int(ij[0]), 5:7] = delta_xy
    return target
