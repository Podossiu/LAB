import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda
import os
PATH = "/data/imagenet"
# 원 핫 인코딩으로 바꿔줌
ds = datasets.ImageFolder(
        root = os.path.join(PATH,"train"),
        transform = ToTensor(),
        target_transform = Lambda(
            # target transform으로 인해 원핫 인코딩으로 변함
            lambda y : torch.zeros(10, dtype = torch.float).scatter_(0, torch.tensor(y), value = 1))
        )
# ToTensor(): PIL Imgage 나 ndarray를 FloatTensor로 변환하며 이미지 픽셀의 크기 값을 
# [0.,1.] 범위로 비례하여 조정함

print(ds[0][1])
