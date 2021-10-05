# SSD_Pytorch_learn
本项目根据ssd.pytorch修改而来，主要用于学习理解SSD目标检测算法。代码中增加大量中文注释帮助理解。

感谢原作者：https://github.com/amdegroot/ssd.pytorch

## 数据集
推荐使用labelimg进行标注。
dataset文件夹下的data存放图片，label存放标注的xml文件。

## 参数
config.py中存放了各种可调节的参数，并给了详细注释。

训练时，需要将all_classes中的标签名替换为自己的标签名，然后执行`python train.py`

## 检测
`python detection.py` 

可在config文件中配置要检测的图片所在文件夹的路径