# 분류기 학습하기
"""
    이미지나 텍스트, 오디오나 비디오 데이터를 다룰 때는 표준 Python 패키지를 이용하여
    Numpy 배열로 불러올 수 있다. 그 후 배열을 torch.*Tensor로 변환하도록 한다.

    --> 이미지 --> Numpy --> Tensor

    -이미지는 Pillow나 OpenCV같은 패키지가 유용하다.
    -오디오는 SciPy와 LibROSA가 유용하다.
    - 텍스트는 그냥 Python이나 Cython을 사용해도 되고 NLTK나 SpaCy도 유용하다.
    특별히 영상 분야를 위한 torchvision이라는 패키지가 만들어져 있는데 
    여기에는 ImageNet이나 CIFAR10, MNIST등 일반적으로 사용하는 데이터셋을 위한
    dataloader, 즉 torchvision.datasets와 data transformer, torch.utils.data.DataLoader가
    포함되어있다.
    매번 유사한 코드를 반복해서 작성하는 것을 피할 수 있도록해준다.
    이 튜토리얼에서는 CIFAR10을 이용하여 분류를 해보도록 하며, 3x32x32로 이루어져 있다.
"""

# 이미지 분류기 학습하기
"""
    1. torchvision을 사용하여 CIFAR10의 학습용/시험용 데이터셋을 불러오고 정규화한다.
    2. CNN을 정의한다.
    3. Loss function을 정의한다.
    4. 신경망을 학습한다.
    5. 신경망을 검사한다.
"""
# 1. CIFAR10을 불러오고 정규화하기

import torch
import torchvision
import torchvision.transforms as transforms

# torchvision의 데이터셋의 output은 [0, 1]의 범위를 갖는 PILImage이다. 이를 [-1,1]의 범위
# 로 정규화된 Tensor로 변환하도록한다.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
batch_size = 4
trainset = torchvision.datasets.CIFAR10(root = "/data",
        train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
        shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root = "/home/ilena7440/pytorch/",
        train = False,download = True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
        shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
        'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 학습용 이미지 보기
import matplotlib.pyplot as plt
import numpy as np

# 이미지를 보여주기 위한 함수

def imshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 학습용 이미지를 무작위로 가져오기
dataiter = iter(trainloader)
images, labels = dataiter.next()
"""
# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))
# 정답 출력
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))
"""
# 신경망 정의하기
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # 배치를 제외하고 flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# GPU로 보내기 
net.to(device)

# loss function, Optimizer 정의하기
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr = 0.001, betas = (0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer = optimizer, step_size = 10, gamma = 0.5)
# 4. 신경망 학습하기
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # [inputs, labels] 목록인 data로 부터 입력받은 후gpu로 보낸다.
        inputs, labels = data[0].to(device), data[1].to(device)

        # 변화도 매개변수를 0으로 만든다.
        optimizer.zero_grad()

        # 순전파 +역전파 + 최적화를 한 후
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 통계를 출력한다.
        running_loss += loss.item()
        if i % 2000 == 1999: # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

"""
    신경망이 예측한 출력과 ground-truth를 비교하는 방식으로 확인하도록 한다.
    만약 예측이 맞다면 correct-predictions 목록에 넣는다.
"""
"""
# 시험용 데이터 확인하기
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
# 이미지를 출력한다.
"""

net = Net()
net.load_state_dict(torch.load(PATH))
net.to(device)
dataiter = iter(testloader)
images, labels = dataiter.next()[0].to(device), dataiter.next()[1].to(device)

outputs = net(images)
# 시험용 데이터로 신경망 검사하기

_, predicted = torch.max(outputs, 1)

print('predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))
correct = 0
total = 0
# 학습중이 아니므로 no_grad를 통해 추적을 멈춤
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # 신경망에 이미지를 통과시켜 출력을 계산한다.
        outputs = net(images)
        # 가장 높은 값을 갖는 분류를 정답으로 선택함
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' %(100 * correct/total))

