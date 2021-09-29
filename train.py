import torch

from SSD import SSD
from torch import optim
from MultiboxLoss import MultiboxLoss
from config import *
from torch.utils.data import DataLoader
from dataset.datasets import MyDataSet
def train():
    ssd_net=SSD(300,all_classes)
    optimizer=optim.SGD(ssd_net.parameters(),lr=LEARNING_RATE,momentum=MOMENTUM,weight_decay=WEIGHT_DECAY)
    criterion=MultiboxLoss(len(all_classes),False)
    ssd_net.train()

    print("Load DataSet ......")
    dateset=MyDataSet()
    dataLoader=DataLoader(dateset,BATCH_SIZE,shuffle=False,collate_fn=batch_collate)
    print("Begin Training ......")
    for k in range(EPOCH):
        total_loss=0
        for data,label in dataLoader:
            clss,locs,prioryBoxes=ssd_net(data)# (batch_size,[10,n_priory,n_classes]),(batch_size,[10,n_priory,4])
            loss_c,loss_l=criterion(clss,locs,prioryBoxes,label)
            loss=loss_l+loss_c
            total_loss+=loss
            optimizer.zero_grad()
            loss.backward(torch.ones_like(loss))  # loss需要是一个标量
            optimizer.step()
        if k%10==0:
            print(f"iter:{k},loss:{total_loss}")
    torch.save(ssd_net.state_dict(),MODEL_SAVE_PATH)

def batch_collate(batch):
    """
    处理batch，解决每张图传入的gt_box数量不一致的问题
    :param batch:
    :return:
    """
    imgs=[]
    targets=[]
    for sample in batch:
        imgs.append(sample[0].unsqueeze(0))
        targets.append(sample[1])
    return torch.cat(imgs,dim=0),targets
if __name__ == '__main__':
    train()
