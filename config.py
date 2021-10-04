
# 生成先验框相关
ASPECT_RATIOS = [[2., 1 / 2], [2., 1 / 2, 3., 1 / 3], [2., 1 / 2, 3., 1 / 3], [2., 1 / 2, 3., 1 / 3], [2., 1 / 2], [2., 1 / 2]]
FEATURE_MAP_SIZE = [38, 19, 10, 5, 3, 1] # 每一层的特征图的大小

# IOU匹配时的阈值
THRESHHOLD=0.5

# 正负样本比例
NEGPOS_RATIO=3.0

# 学习率
LEARNING_RATE=0.01

# 随机梯度下降时的超参数
MOMENTUM=0.9
WEIGHT_DECAY=0.5e-4  # 防止过拟合，调节复杂度对模型损失函数的影响，若weight decay很大，则复杂的模型损失函数的值也就大

# BATCH
BATCH_SIZE = 64
# EPOCH
EPOCH = 100

all_classes=[
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

# 模型保存位置
MODEL_SAVE_PATH="test.pth"

# 要预测的图片保存的位置
PRED_PIC_PATH="dataset/pred"
# 预测结果输出位置
PRED_PIC_OUTPUT="output"

# 非极大值抑制算法相关参数
NMS_THRESHHOLD=0.5
NMS_TOP_K=200

# 预测时置信度的最小值
PRED_MIN_THRESHHOLD=0.9