# 신경망 정의하기
"""
    신경망은 torch.nn 패키지를 이용하여 생성할 수 있따.
    지금까지 autograd를 살펴봣는데, nn은 모델을 정의하고 미분하는데 autograd를 사용한다.
    nn.Module은 계층과 output을 반환하는 forward(input) 메서드를 포함하고 있다.
    LeNet을 살펴보면 간단한 순전파 네트워크이다. input을 받아 여러 계층에 차례로
    전달한 뒤, 최종 출력을 제공한다.

    신경망의 일반적인 학습 과정은 다음과 같다.
    - 학습 가능한 매개변수를 갖는 신경망을 정의한다.
    - 데이터셋 입력을 반복한다.
    - 입력을 신경망에서 전파한다.
    - loss를 계산한다.
    - gradient를 신경망의 매개변수들에 역으로 전파한다.
    - 신경망의 가중치를 갱신한다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 입력 이미지 채널 1개, 출력 채널 6개, 5 x 5의 정사각 Conv 행렬
        # Conv 커널 정의
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6,16, 5)
        # affine 연산
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # (2, 2)크기 윈도우에 대해 max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        # 크기가 하나의 square라면 하나의 숫자만을 ㅈ특정
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = torch.flatten(x, 1) # 배치 차원을 제외하고 flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
print(net)

"""
    forward 함수만 정의하고 나면 backward함수는 autograd를 통하여 자동으로 정의된다.
    forward함수에서는 어떠한 Tensor연산을 사용해도 된다.
    모델의 학습 간으한 매개변수들은 net.parameters()에 의해 반환된다.
"""
params = list(net.parameters())
print(len(params))
print(params) # conv1의 .weight

# 임의의 32x32 입력값을대입한다.
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# 모든 매개변수의 grad buffer를 0으로 설정하고 무작위 값으로 역전파한다.
net.zero_grad()
out.backward(torch.randn(1, 10))

""" 
    torch.nn은 미니배치만 지원한다. torch.nn 패키지 전체는 하나의 샘플이 아닌,
    샘플들의 미니배치만을 입력으로 받는다. 예를들어, nnConv2d는 
    nSamples x nChannels x Height x Width의 4차원 Tensor를 입력으로한다.
    만약 하나의 샘플만 있다면 input.unsqueeze(0)을 사용하여 가상의 차원을 추가한다.

    요약
    - Torch.Tensor - backward()와 같은 autograd 연산을 지원하는 다차원배열이다.
    - nn.Module - 신경망 모듈, 매개변수를 캡슐화하는 간편한 방법으로, 
    GPU로 이동, 내보내기, 불러오기 등의 작업을 위한 helper를 제공한다.
    - nn.Prameter - Tensor의 한 종류로, Module에 속성으로 할당될 때 자동으로 
    매개변수에 등록된다.
    - autograd.Function - autograd연산의 순방향과 역방향 정의를 구현한다.
    Tensor연산은 하나 이상의 Function 노드를 생성하며 각 노드는 Tensor를 생성하고
    이력을 인코딩하는 함수들과 연결하고 있다.
"""
# 손실 함수
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0]) # linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0]) # ReLU

# 역전파
"""
    error를 역전파하기위해서는 loss.backward()만 해주면된다. 기존에 계산된 
    변화도의 값을 누적시키고 싶지 않다면, 기존에 계산된 변화도를 0으로 만드는 작업이
    필요합니다.
"""
net.zero_grad()
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 가중치 갱신
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr = 0.01)
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step() # 업데이트 진행

