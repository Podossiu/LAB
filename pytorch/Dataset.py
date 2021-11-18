# 데이터셋 로드
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib
from torch.utils.data import Dataset, DataLoader
import os

PATH="/data/imagenet"

training_data = datasets.ImageFolder(
        root = os.path.join(PATH, "train"),
        transform = ToTensor(),
        )
test_data = datasets.ImageFolder(
        root = os.path.join(PATH, "validation"),
        transform = ToTensor(),
        )
# 데이터셋 시각화
figure = plt.figure(figsize=(8,8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img,label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img[1,:,:])
plt.show()


