# 신경망 모델 구성하기
import os 
import torch
from torch import nn # torch.nn 네임스페이스는 신경망을 구성하는데 필요한 모든 요소들
                     # 을 제공함 Pytorch의 모든 모듈은 nn.Module의 하위 클래스임
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 학습을 위한 장치얻기
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# 클래스 정의하기
# 신경망 모델을 nn.Module의 하위 클래스로 정의, __init__에서 신경망 계층들을 초기화함
# nn.Module을 상속받은 모든 클래스는 forward 메소드에 입력 데이터에 대한 연산을 구현

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10),
                nn.ReLU()
                )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# NeuralNetwork의 인스턴스를 생성하고 device로 이동, structure를 출력
model = NeuralNetwork().to(device)
print(model)

# 모델을 사용하기 위해 입력 데이터를 전달, 일부 백그라운드 연산들과 함께 모델의 forward를 실행함
# model.forward()를 직접 호출하지 말것
X = torch.rand(1, 28, 28, device = device)
logits = model(X)
pred_prob = nn.Softmax(dim=1)(logits)
y_pred = pred_prob.argmax(1)
print(f"Predicted class: {pred_prob}, {y_pred}")

# 모델 계층 ( Layer )
# FashionMNIST 모델의 계층을 살펴본다, 이를 설명하기 위해 28x28 크기의 이미지 3개로 구성된 미니 배치를 가져온다.
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# nn.Flatten
# nn.Flatten 계층을 초기화하여 각 28x28의 2D 이미지를 784 픽셀 값을 갖는 연속된 배열로 변환하도록 한다. ( dim = 0 는 미니배치 차원 )
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# nn.Linear 
# 선형 계층은 저장된 weight와 bias를 사용하여 입력에 선형 변환을 적용하는 모듈이다.
layer1 = nn.Linear(in_features= 28 * 28, out_features = 20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# nn.ReLU 
# 비선형 활성화는 모델의 입력과 출력 사이에 복잡한 관계 ( mapping )을 만든다. 비선형 활성화는 선형 변환 후에 적용되어 비선형성을 도입하고 신경망이 다양한 현상을 학습할 수 있도록 돕는다.

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# nn.Sequential
# nn.Sequential은 순서를 갖는"모듈의 컨테이너"이다.데이터는 정의된 것과 같은 순서로 모든 모듈들을 통해 전달된다. 순차 컨테이너를 사용하여 아래의 seq_module과 같은 신경망을 빠르게 만들 수 있다.

seq_modules = nn.Sequential(
        flatten,
        layer1,
        nn.ReLU(),
        nn.Linear(20, 10)
        )
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

# nn.Softmax
# 신경망의 마지막 선형 계층은 nn.softmax 모듈에 전달될 [ -inf, inf ]범위의 원시값인 logis를 반환한다. logits는 모델의 각 분류에 대한 예측 확률을 나타내도록 [0,1]범위로 비례하여 조정된다. dim 매개변수는 값의 합이 1이 되는 차원을 나타낸다.
softmax = nn.Softmax(dim = 1)
pred_prob = softmax(logits)

# 모델 매개변수
#신경망 내부의 많은 계층들은 매개변수화된다. 즉 학습 중에 최적화되는 가중치와 편향에 연관지어진다.
# nn.Module을 상속하면 모델 객체 내부의 모든 필드들이 자동으로 추적되며, 모델의 parameters() 및 named_parameters()메소드로 모든 매개변수에 접근할 수 있게 된다.

print("Model structure : ", model, "\n\n")
for name, param in model.named_parameters():
    print(f"Layer : {name} | Size : {param.size()} | Values : {param[:2]} \n")

