import torch
from torch import nn
from config import *
from torch.autograd import Variable


class MultiboxLoss(nn.Module):
    def __init__(self, num_classes: int, use_gpu: bool):
        super(MultiboxLoss, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu

    def forward(self, pred_conf: torch.Tensor, pred_loc: torch.Tensor, priory_boxes: torch.Tensor, truth: torch.Tensor):
        """
        :param pred_conf: 预测的置信度，(batch_size,n_priory,n_classes)
        :param pred_loc: 预测的位置, (batch_size,n_priory,4)
        :param priory_boxes: 先验框,(n_priory,4)
        :param truth: 真值框及其类别(batch_size,n_objects,5) 第五个数是类别
        :return: 返回损失值
        """
        num = pred_loc.size(0)  # batch大小
        num_priroy = priory_boxes.size(0)
        num_classes = self.num_classes

        conf_t = torch.Tensor(num, num_priroy)  # 存放的是每个先验框对应的目标物体的序号，0表示背景
        loc_t = torch.Tensor(num, num_priroy, 4)
        for idx in range(num):
            # 对每个数据操作,为每一个先验框匹配一个真值框
            self.match(truth[idx, :, :4], priory_boxes, truth[idx, :, 4], loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # 包装变量
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)
        # 选出非背景框，即IOU>0的框
        pos = conf_t > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(
            loc_t)  # (batch_size,num_priory)->(batch_size,num_priory,1)->(batch_size,num_priory,4)
        # 获取这些框的坐标
        loc_pred = pred_loc[pos_idx].view(-1, 4)  # (num_pos,4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        # 计算位置损失
        loss_loc = self.smooth_L1_loss(loc_pred, loc_t)

        batch_conf = pred_conf.view(-1, self.num_classes)  # (batch_size*num_classes,n_classes)
        # 参考 https://www.freesion.com/article/9127332548/
        loss_conf = self.log_sum_exp(batch_conf) - \
                    batch_conf.gather(dim=1, index=conf_t.view(-1, 1).long())  # 第k行的第i列，由原始数据的第k行的index[k][i]列决定
        # Hard Negative Mining
        loss_conf=loss_conf.view(num,-1)
        loss_conf[pos] = 0  # 过滤掉正样本
        loss_conf = loss_conf.view(num, -1)  # (batch_size, n_priory)
        # 一次sort：得到的index是按顺序排的索引
        # 两次sort：得到原Tensor的映射，排第几的数字变为排名
        _, loss_idx = loss_conf.sort(1, descending=True)  # 将loss按照从大到小排序
        _, idx_rank = loss_conf.sort(1)  # 将loss从大到小排序返回的序号按照列优先进行排序
        num_pos = pos.long().sum(1, keepdims=True)  # 计算正样本数
        num_neg = torch.clamp(NEGPOS_RATIO * num_pos, max=num_pos.size(1) - 1)  # 负样本数
        neg = idx_rank < num_neg.expand_as(idx_rank)  # 筛选出排名小于负样本数的样本，结果中为True的为筛选出的样本

        pos_idx = pos.unsqueeze(2).expand_as(pred_conf)
        neg_idx = neg.unsqueeze(2).expand_as(pred_conf)
        # conf_p为选出的正样本和负样本
        conf_p = pred_conf[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)  # gt(0)大于0则为1，小于0则为0
        targets_weighted = conf_t[(pos + neg).gt(0)]
        # 参考 https://www.cnblogs.com/marsggbo/p/10401215.html
        loss_c = nn.functional.cross_entropy(conf_p, targets_weighted.long(), size_average=False)  # 交叉熵计算

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        loss_loc /= N
        loss_c /= N
        return loss_loc, loss_c

    def match(self, truth_boxes: torch.Tensor, priory_boxes: torch.Tensor, labels: torch.Tensor, loc_t: torch.Tensor,
              conf_t: torch.Tensor, idx):
        """
        匹配先验框与GT，使得每一个先验框都有一个label
        :param truth_boxes: [n_obj,4]
        :param priory_boxes: [n_priory_boxes,4]
        :param labels: [n_obj]
        :param loc_t: 坐标结果
        :param conf_t: 置信度结果
        :param idx: 上述结果的index
        :return:
        """
        IOU = self.calculateIOU(self.transFromBoxToPoint(priory_boxes), truth_boxes)  # 计算重叠部分比例, (n_priroy,n_boxes_gt)
        # 为每一个gt框匹配一个最大IOU的先验框
        best_priory_overlap, best_priory_idx = torch.max(IOU, dim=0, keepdim=True)
        # 为每一个先验框匹配一个gt框
        best_gt_overlap, best_gt_idx = torch.max(IOU, dim=1, keepdim=True)

        best_gt_overlap.squeeze_(0)
        best_gt_idx.squeeze_(0)
        best_priory_overlap.squeeze_(1)
        best_priory_idx.squeeze_(1)

        # 对于一个gt框对应的最佳先验框，置其IOU值为2,表示最佳匹配
        best_gt_overlap.index_fill(0, best_priory_idx, 2)  # 用2填充，表示最佳先验框
        for j in range(best_priory_idx.size(0)):
            best_gt_idx[best_priory_idx[j]] = j  # j为gt框编号，best_priory_idx[j]为先验框编号

        # 由于传入进来的labels的类别是从0开始的，SSD中认为0应该是背景，所以，需要对labels进行加一
        conf = torch.Tensor([labels[best_gt_idx[i]] + 1 for i in range(best_gt_idx.size(0))]).view(-1,1)
        conf[best_gt_overlap < THRESHHOLD] = 0  # 小于阈值的设置为0,conf里存的是每个先验框对应的标签

        # 取出每一个先验框对应的最佳GT框的坐标
        matched = truth_boxes[best_gt_idx].view(-1,4)

        # 对坐标进行编码
        matched_encoded = self.encode(matched, priory_boxes)
        # 存储结果
        conf_t[idx] = conf.view(-1)
        loc_t[idx] = matched_encoded

    def encode(self, matched_boxes: torch.Tensor, priory_boxes: torch.Tensor):
        """
        利用GT框和先验锚点框,计算偏差,用于回归
        :param matched_boxes: 每一个先验框对应的最佳GT框的坐标，shape:(n_priory,4) 坐标为点的形式
        :param priory_boxes: 先验框，坐标为点加长宽的形式
        :return: 返回编码后的坐标，shape:(n_priory,4)
        """
        l_cxcy = (matched_boxes[:, :2] + matched_boxes[:, 2:]) / 2 - priory_boxes[:, :2]  # 先将matched_boxes转换为中心点坐标
        l_cxcy = l_cxcy / priory_boxes[:, 2:]

        l_wh = torch.log((matched_boxes[:, 2:] - matched_boxes[:, :2]) / priory_boxes[:, 2:])

        return torch.cat([l_cxcy, l_wh], dim=1)  # 按照列拼接,拓展列

    def transFromBoxToPoint(self, boxes: torch.Tensor):
        """
        将(x_ctr,y_ctr,w,h)表示的anchor转化为(x,y,x,y)
        :param boxes: (n_boxes,4)
        :return: (n_boxes,4)
        """
        newBoxes = torch.Tensor(boxes.shape)
        newBoxes[:, 0] = boxes[:, 0] - 0.5 * boxes[:, 2]
        newBoxes[:, 1] = boxes[:, 1] - 0.5 * boxes[:, 3]
        newBoxes[:, 2] = boxes[:, 0] + 0.5 * boxes[:, 2]
        newBoxes[:, 3] = boxes[:, 1] + 0.5 * boxes[:, 3]
        return newBoxes

    def calculateIOU(self, boxes_priory: torch.Tensor, boxes_gt: torch.Tensor) -> torch.Tensor:
        """
        计算boxes_priory与boxes_gt的iou
         A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        :param boxes_priory: shape: (8732,4)
        :param boxes_gt: shape: (n_boxes_gt,4)
        :return: shape: (8732,n_boxes_gt)
        """
        A = boxes_priory.size(0)
        B = boxes_gt.size(0)
        # 计算两部分的面积和
        size_priory = (boxes_priory[:, 2] - boxes_priory[:, 0]) * (boxes_priory[:, 3] - boxes_priory[:, 1])
        size_gt = (boxes_gt[:, 2] - boxes_gt[:, 0]) * (boxes_gt[:, 3] - boxes_gt[:, 1])
        size_total = size_priory.unsqueeze(1).expand(A, B) + size_gt.unsqueeze(0).expand(A, B)  # (8732,n_boxes_gt)
        # 计算交叉部分的面积
        max_xy = torch.min(boxes_priory[:, 2:].unsqueeze(1).expand(A, B, 2),
                           boxes_gt[:, 2:].unsqueeze(0).expand(A, B, 2))  # 右下角坐标的较小值
        min_xy = torch.max(boxes_priory[:, :2].unsqueeze(1).expand(A, B, 2),
                           boxes_gt[:, :2].unsqueeze(0).expand(A, B, 2))  # 左上角坐标的较大值
        clamp = torch.clamp(max_xy - min_xy, min=0)  # 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量。此处无需限制max
        size_cross = clamp[:, :, 0] * clamp[:, :, 1]
        return size_cross / size_total - size_cross

    def smooth_L1_loss(self, y_pred: torch.Tensor, y_true):
        """
        :param y_pred: (batch_size,n_boxes,4)
        :param y_true: (batch_size,n_boxes,4)
        :return: (batch_size,n_boxes)
        """
        abs_y_loss = torch.abs(y_pred - y_true)
        square_loss = torch.pow(abs_y_loss, 2)
        loss_index_part1 = torch.where(abs_y_loss < 1.0)
        loss_index_part2 = torch.where(abs_y_loss >= 1.0)
        loss = torch.Tensor(y_pred.shape)
        loss[loss_index_part1] = square_loss[loss_index_part1] * 0.5
        loss[loss_index_part2] = abs_y_loss[loss_index_part2] - 0.5
        return torch.sum(loss, dim=-1)  # 最后一维


    def log_sum_exp(self,x:torch.Tensor):
        x_max=x.detach().max()
        return torch.log(torch.sum(torch.exp(x-x_max),1,keepdim=True))+x_max

