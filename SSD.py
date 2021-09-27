import torch
from torch import nn
from prioryBox import generate_priory_box
from L2Norm import L2Norm
from config import *


class SSD(nn.Module):
    def __init__(self, img_size: int, cls: list):
        super(SSD, self).__init__()
        self.objClasses = cls
        # conv1-4
        self.VGG16_before = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), 1),
            nn.ReLU()
        )
        # conv5
        self.VGG16_after = nn.Sequential(
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(512, 512, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), 1),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=1, padding=1)
        )
        self.conv6 = nn.Sequential(nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=6, dilation=(6, 6)),
                                   nn.ReLU())  # 19+6*2-((6-1)*2+3)+1=19 #空洞卷积，增大感受野
        self.conv7 = nn.Sequential(nn.Conv2d(1024, 1024, (1, 1), (1, 1)), nn.ReLU())  # 19
        self.conv8 = nn.Sequential(nn.Conv2d(1024, 256, (1, 1), (1, 1)), nn.ReLU(),
                                   nn.Conv2d(256, 512, (3, 3), (2, 2), 1), nn.ReLU())  # (19+1*2-(3-2))/2=10
        self.conv9 = nn.Sequential(nn.Conv2d(512, 128, (1, 1), (1, 1)), nn.ReLU(),  # 10
                                   nn.Conv2d(128, 256, (3, 3), (2, 2), 1), nn.ReLU())  # (10+1*2-(3-2))/2=5.5~5
        self.conv10 = nn.Sequential(nn.Conv2d(256, 128, (1, 1), (1, 1)), nn.ReLU(),
                                    nn.Conv2d(128, 256, (3, 3), (2, 2), 1), nn.ReLU())  # (5+1*2-(3-2))/2=3
        self.pool11 = nn.AvgPool2d(3)

        # L2正则化
        self.L2Norm = L2Norm(512, 20)

        # priory box
        self.img_size = img_size
        self.priory_boxes = generate_priory_box(self.img_size)  # 生成先验框

        # loc 3*3卷积生成位置
        self.locConv = []
        self.locConv.append(
            nn.Conv2d(512, (2 + len(ASPECT_RATIOS[0])) * 4, (3, 3), padding=1))
        self.locConv.append(
            nn.Conv2d(1024, (2 + len(ASPECT_RATIOS[1])) * 4, (3, 3), padding=1))
        self.locConv.append(
            nn.Conv2d(512, (2 + len(ASPECT_RATIOS[2])) * 4, (3, 3), padding=1))
        self.locConv.append(
            nn.Conv2d(256, (2 + len(ASPECT_RATIOS[3])) * 4, (3, 3), padding=1))
        self.locConv.append(
            nn.Conv2d(256, (2 + len(ASPECT_RATIOS[4])) * 4, (3, 3), padding=1))
        self.locConv.append(
            nn.Conv2d(256, (2 + len(ASPECT_RATIOS[5])) * 4, (3, 3), padding=1))

        # cls 3*3卷积生成分类
        self.clsConv = []
        self.clsConv.append(
            nn.Conv2d(512, (2 + len(ASPECT_RATIOS[0])) * len(self.objClasses), (3, 3),
                      padding=1))
        self.clsConv.append(
            nn.Conv2d(1024, (2 + len(ASPECT_RATIOS[1])) * len(self.objClasses), (3, 3), padding=1))
        self.clsConv.append(
            nn.Conv2d(512, (2 + len(ASPECT_RATIOS[2])) * len(self.objClasses), (3, 3), padding=1))
        self.clsConv.append(
            nn.Conv2d(256, (2 + len(ASPECT_RATIOS[3])) * len(self.objClasses), (3, 3), padding=1))
        self.clsConv.append(
            nn.Conv2d(256, (2 + len(ASPECT_RATIOS[4])) * len(self.objClasses), (3, 3), padding=1))
        self.clsConv.append(
            nn.Conv2d(256, (2 + len(ASPECT_RATIOS[5])) * len(self.objClasses), (3, 3), padding=1))

    def forward(self, x):
        sources = []  # sources为每个层计算出的feature_map
        # 计算到conv4
        conv4 = self.VGG16_before(x)
        s = self.L2Norm(conv4)
        sources.append(s)

        # 计算到conv7
        vgg16_after=self.VGG16_after(s)
        conv6 = self.conv6(vgg16_after)
        conv7 = self.conv7(conv6)
        sources.append(conv7)

        # 计算到conv8
        conv8 = self.conv8(conv7)
        sources.append(conv8)

        # 计算到conv9
        conv9 = self.conv9(conv8)
        sources.append(conv9)

        # 计算到conv10
        conv10 = self.conv10(conv9)
        sources.append(conv10)

        # 计算到pool11
        pool11 = self.pool11(conv10)
        sources.append(pool11)

        # 根据获取到的feature map 生成分类cls，和目标框loc
        clss=[]
        locs=[]
        for f_map,l_conv,c_conv in zip(sources,self.locConv,self.clsConv):
            clss.append(c_conv(f_map).permute(0,2,3,1).contiguous())# 最后一维为分类数
            # contiguous()方法首先拷贝了一份张量在内存中的地址，然后将地址按照形状改变后的张量的语义进行排列
            locs.append(l_conv(f_map).permute(0,2,3,1).contiguous())

        # 转化为Tensor类型
        pred_conf=torch.cat([o.view(o.size(0),-1,len(self.objClasses)) for o in clss],dim=1)  # 转化为(batch_size,[10,n_priory,n_classes])
        pred_loc=torch.cat([o.view(o.size(0),-1,4)for o in locs],dim=1)  # 转化为(batch_size,[10,n_priory,4])
        # print(pred_loc.size(),pred_conf.size())
        return pred_conf,pred_loc,self.priory_boxes





