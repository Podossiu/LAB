# 사용자 정의 데이터셋
import os 
import pandas as pd
from torchvision.io import read_image

# 데이터셋 클래스는 3개의 함수 구현
# __init__, __len__, __getitem__
# 이미지들은 img_dir에저장, 정답은 annotations_file에 저장
class CustomImageDataset(Dataset):
    # 두가지변형 초기화 + 이미지와 주석 파일 디렉토리 초기화
    def __init__(self, annotations_file, img_dir, transform=None,
            target_transform= None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    # 데이터셋의 샘플 갯수 초기화
    def __len__(self):
        return len(self.img_labels)
    # idx에 해당하는 데이터 샘플 불러오고 반환 
    # 1. 인덱스를 기반으로 디스크에서 이미지의 위치 식별
    # 2. csv데이터롣부터 정답을 반환
    # 3. ( 해당하는 경우 transform 함수 호출 )
    # 4. 텐서 이미지와 라벨을 dict형으로 반환
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image" : image, "label":label}
        return sample


