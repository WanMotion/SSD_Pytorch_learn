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
    dataLoader=DataLoader(dateset,BATCH_SIZE,shuffle=True)
    print("Begin Training ......")
    for k in range(EPOCH):
        total_loss=0
        for data,label in dataLoader:
            print(data.size(),label.size())
            clss,locs,prioryBoxes=ssd_net(data)
            loss=criterion(clss,locs,prioryBoxes,label)
            total_loss+=loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if k%10==0:
            print(f"iter:{k},loss:{total_loss}")
    torch.save(ssd_net.state_dict(),MODEL_SAVE_PATH)

if __name__ == '__main__':
    train()
