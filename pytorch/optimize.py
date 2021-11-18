import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

training_data = datasets.FashionMNIST(
        root = "/home/ilena7440/pytorch",
        train = True,
        download = True,
        transform = ToTensor()
        )

test_data = datasets.FashionMNIST(
        root = "/home/ilena7440/pytorch",
        train = False,
        download = True,
        transform = ToTensor()
        )

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512,512),
                nn.ReLU(),
                nn.Linear(512,10),
                nn.ReLU()
                )

    def forward(self,x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()

# hyperparameter
"""
   서로 다른hyperparameter값은 모델 학습과 수렴율에 영향을 미칠 수 있다.
   1. epoch
   2. batch size
   3. learning rate
"""

learning_rate = 1e-3
batch_size = 64
epochs = 5

# 최적화 단계 
""" 
    hyperparameter를 설정한 뒤에는 최적화 단계를 통해 모델을 학습하며
    최적화할 수 있다. 각 단계의 반복을 epoch으로 구성된다.
    하나의 에폭은 다음 두 부분으로 구성된다.
    1. 학습 단계 ( train loop )
    2. 검증 / 테스트 단계 ( validation loop )
"""

# 손실 함수 ( loss function )
"""
    손실함수는 획득한 결과와 실제값 사이의 틀린 정도를 측정하며, 학습 중에 이를 최소화
    하려 한다. 주어진 데이터 샘플을 입력으로 계산한 예측과 정답을 비교하여 손실을 계산
    한다.
    손실함수에는 regression에 사용하는 nn.MSELoss, classification에 사용하는 nn.NLLLoss     ( Negative Log Likelihood ) 그리고 nn.LogSoftmax와 nn.NLLLoss를 합친
    nn.CrossEntropyLoss등이 있다.
"""
loss_fn = nn.CrossEntropyLoss()

# Optimizer
"""
    SGD를 정의한다. 모든 최적화 logic은 optimizer 객체에 캡슐화된다.
    여기서는 SGD 옵티마이저를 사용하고 있으며 Pytorch에는 ADAM이나 RMSProp과 같은
    다른 종류의 모델과 데이터에서 더 잘 동작하는 다양한 옵티마이저가 있다.
    학습하려는 모델의 매개변수와 learning_rate를 등록하여 옵티마이저를 초기화한다.
"""
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

"""
    학습 단계에서 최적화는 세단계로 이루어진다.
    1. optimizer.zero_grad()를 호출하여 모델 매개변수의 변화도를 재설정한다.
    기본적으로 변화도는 더해지기 때문에 중복 계산을 막기 위해 반복할 때마다 명시적으로
    0으로 설정한다.

    2. loss.backward()를 호출하여 예측 손실 ( prediction loss )를 역전파한다.
    PyTorch는 각 매개변수에 대한 손실의 변화도를 저장한다.

    3. 변화도를 계산한 뒤에는 optimizer.step()을 호출하여 역전파 단계에서 수집된
    변화도로 매개변수를 조정한다.
"""

# 전체 구현
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # 예측과 손실 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}\n")

# 손실함수와 옵티마이저를 초기화하며 train_loop와 test_loop에 전달한다. 모델의 성능 향상을 알아보기 위해 자유롭게 에폭 수를 증가시켜 볼 수 있다.

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate * 3)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer = optimizer,
                                             lr_lambda = lambda epoch: 0.95 ** epoch,
                                             last_epoch = -1,
                                             verbose = False
                                             )
epochs = 20
for t in range(epochs):
    print(f"Epoch {t+1}\n------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")


