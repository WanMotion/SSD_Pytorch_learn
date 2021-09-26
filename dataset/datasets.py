import torch
from torch.utils.data import Dataset
import os
import xml.etree.ElementTree as ET
from config import *
from PIL import Image
from torchvision import transforms
from pretreatment import *

LABEL_PATH = "./dataset/label"


class MyDataSet(Dataset):
    def __init__(self):
        self.labels_files = [os.path.join(LABEL_PATH, file) for file in os.listdir(LABEL_PATH)]

    def __getitem__(self, index):
        tree = ET.parse(self.labels_files[index])
        root = tree.getroot()
        res = []
        for obj in root.iter('object'):
            xml_box = obj.find('bndbox')
            # 读取相应坐标
            xmin = (int(xml_box.find('xmin').text))
            ymin = (int(xml_box.find('ymin').text))
            xmax = (int(xml_box.find('xmax').text))
            ymax = (int(xml_box.find('ymax').text))
            # 读取类别
            cls = obj.find('name').text
            idx = all_classes.index(cls)
            assert idx >= 0
            res += [xmin, ymin, xmax, ymax, idx]
        # 读取图片
        img_path = root.find('folder').text + '/' + root.find('filename').text
        img = Image.open(img_path)
        # 图像预处理
        transform_resize=Resize()
        img=transform_resize(img)
        # 转换为Tensor
        toTensor = transforms.ToTensor()

        return toTensor(img), torch.Tensor(res).view((-1,5))

    def __len__(self):
        return len(self.labels_files)
