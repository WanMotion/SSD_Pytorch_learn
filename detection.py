import os

import torch

from SSD import SSD
from config import *
from dataset.datasets import DetectionDataset
from torch.utils.data import DataLoader
import cv2

def pred():
    SSD_net = SSD(300, all_classes)
    SSD_net.load_state_dict(torch.load(MODEL_SAVE_PATH))

    detectionData = DetectionDataset()
    dataloader = DataLoader(detectionData, BATCH_SIZE, collate_fn=bacth_dealer)

    SSD_net.eval()
    for d in dataloader:
        imgs, paths = d
        pred_conf, pred_loc, _ = SSD_net(imgs)  # (batch_size,n_priory,n_classes), (batch_size,n_priory,4)
        conf_softmax = torch.nn.functional.softmax(pred_conf, dim=-1)
        conf_softmax_max, conf_softmax_idx = torch.max(conf_softmax, dim=-1)  # (batch_size,n_priory)
        conf_softmax_idx=conf_softmax_idx+1
        # 对每张图片做nms
        for i in range(conf_softmax_max.size(0)):
            img_i=cv2.imread(paths[i])
            width=img_i.shape[0]
            height=img_i.shape[1]
            for j in range(1,len(all_classes)+1):
                # 选出每个类的conf和loc
                conf=conf_softmax_max[i,conf_softmax_idx==j]
                loc=pred_loc[i,conf_softmax_idx==j,:]
                keep,count=nms(conf,loc)
                for k in range(count):
                    if conf[keep[k]]>PRED_MIN_THRESHHOLD:
                        img_i=cv2.rectangle(img_i,(loc[keep[k],0]*width,loc[keep[k],1])*height,(loc[keep[k],2]*width,loc[keep[k],3]*height),(255,0,0),thickness=4)
                        img_i=cv2.putText(img_i,f"{all_classes[j-1]}:{conf[keep[k]]}",(loc[keep[k],0]*width,loc[keep[k],1]*height),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0))
            cv2.imwrite(os.path.join(PRED_PIC_OUTPUT,os.path.basename(paths[i])),img_i)









def bacth_dealer(batch):
    imgs = []
    paths = []
    for sample in batch:
        imgs.append(sample[0].unsqueeze(0))
        paths.append(sample[1])
    return torch.cat(imgs, dim=0), paths


def nms(pred_conf: torch.Tensor, pred_loc: torch.Tensor):
    """
    非极大值抑制算法
    :param pred_conf: shape:[n_priory]
    :param pred_loc: shape:[n_priory,4]
    :return:
    """
    pred_conf_sort,max_idx=torch.sort(pred_conf)
    max_idx=max_idx[-NMS_TOP_K:]
    keep = torch.zeros((pred_conf.size(0)), dtype=torch.long)
    if pred_loc.numel() == 0:
        return keep
    x1 = pred_loc[:, 0]
    y1 = pred_loc[:, 1]
    x2 = pred_loc[:, 2]
    y2 = pred_loc[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    count = 0
    while max_idx.numel() > 0:
        i = max_idx[-1]
        keep[count] = i
        count += 1
        max_idx = max_idx[:-1]
        if max_idx.size(0) == 1:
            break
        xx1 = x1[max_idx]
        yy1 = y1[max_idx]
        xx2 = x2[max_idx]
        yy2 = y2[max_idx]
        # 计算IOU，和MultiBox中基本一致
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w = xx2 - xx1
        h = yy2 - yy1
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        iou = inter / (area[max_idx] + area[i] - inter)
        max_idx = max_idx[iou.le(NMS_THRESHHOLD)]
    return keep, count
