import math
from itertools import product
import torch

from config import *


def generate_priory_box(img_size=300):
    """
    生成先验框，参考https://zhuanlan.zhihu.com/p/95032060的讲解，以及https://www.cnblogs.com/beiyi-zyw/p/12067693.html
    :param img_size: 默认为300
    :return: 返回生成的先验框 [A,4] : [x_ctr,y_ctr,w,h]
    """
    min_size = [30]
    max_size = [60]
    min_size += [300 * (0.2 + (0.9 - 0.2) / (5) * (k - 1)) for k in range(1, len(FEATURE_MAP_SIZE))]
    max_size += [300 * (0.2 + (0.9 - 0.2) / (5) * (k - 1)) for k in range(2, len(FEATURE_MAP_SIZE)+1)]
    boxes = []
    for k, size in enumerate(FEATURE_MAP_SIZE):
        for i, j in product(range(size), repeat=2):
            x_ctr = (i + 0.5) / FEATURE_MAP_SIZE[k]
            y_ctr = (i + 0.5) / FEATURE_MAP_SIZE[k]
            boxes += [x_ctr, y_ctr, min_size[k] / img_size, min_size[k] / img_size]
            boxes += [x_ctr, y_ctr, math.sqrt(min_size[k] * max_size[k]) / img_size,
                      math.sqrt(min_size[k] * max_size[k]) / img_size]
            for ratio in ASPECT_RATIOS[k]:
                boxes += [x_ctr, y_ctr, min_size[k] / math.sqrt(ratio) / img_size,
                          min_size[k] * math.sqrt(ratio) / img_size]
    return torch.Tensor(boxes).view(-1, 4)
