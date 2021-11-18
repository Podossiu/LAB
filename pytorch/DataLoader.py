# 데이터 로더로 학습용 데이터 준비하기
# dataset은 데이터셋의 feature를 가져오며, 하나의 샘플에
# 정답을 지정하는 일을 한번에 함
# 모델을 학습할 때 일반적으로 sample들을 minibatch로 전달하며
# 매 epoch마다 데이터를 다시 섞어서 overfit을 막고 
# python의 multiprocessing을 이용하여 데이터 검색 속도를 높임
# DataLoader는 iterable한 객체

from torch.utils.data import DataLoader
import os
import pandas as pd
from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import ToTensor
class Example_Dataset(nn.Module):
    def __init__(self, df, img_folder_path, transforms = None):
        self.images = glob.glob('your_folder_path')
        self.label = df['label'].values

PATH = "/data/imagenet"


train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
test_dataloader = DataLoader(test_data, batch_size = 64, shuffle = True)

# 데이터 로더를 통해 순회하기 ( iterate )
train_features, train_labels = iter(train_dataloader)
print(f"Feature batch shape : {train_features.size()}")
print(f"Labels batch shape : {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img[0,:,:], cmap='gray')
plt.show()
print(f"Label: {label}")
