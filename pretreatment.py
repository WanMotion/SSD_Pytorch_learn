# 预处理图片
from PIL import Image
class Resize(object):
    def __init__(self,size=300):
        self.size=size

    def __call__(self, img:Image):
        return img.resize((self.size,self.size))