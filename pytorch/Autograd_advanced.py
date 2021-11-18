# autograd에 대한 심화적인 내용
# Pytorch에서 사용법
"""
    torchvision에서 pre-trained된 resnet18 모델을 불러온다.
    C = 3, H,W = 64 인 무작위 텐서 생성, label무작위로 초기화
"""
import torch, torchvision
from torch import nn, optim
model = torchvision.models.resnet18(pretrained=True)
data = torch.rand(1,3,64,64)
labels = torch.rand(1,1000)

# 입력 데이터를 각 층에 통과시켜 예측값 생성 ( 순전파 )
prediction = model(data)

# 모델의 예측값과 그에 해당하는 label을 이용하여 오차를 계산한다.
"""
    에러를 역전파하도록 한다. 오차 tensor ( error tensor )에 대해 .backward()를 
    호출하면 역전파가 시작된다. 그 다음 Autograd가 매개변수의 .grad 속성에 모델의
    각 매개변수에 대한 gradient를 계산하고 저장한다.
"""
loss = (prediction - labels).sum()
loss.backward() # 역전파 단계

# optimizer 생성한다. lr = 0.01, momentum = 0.9인 SGD
optim = torch.optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9)
# .step 호출, 경사하강법 시작 
optim.step()

# autograd에서의 미분
"""
    autograd가 어떻게 변화도를 수집하는지 알아본다. requires_trad=True를 갖는 
    2개의 tensor a,b를 만든다. requires_grad=True는 autograd에 모든 연산들을
    추적해야한다는 것을 알려준다.
"""
a = torch.tensor([2., 3.], requires_grad = True)
b = torch.tensor([6., 4.], requires_grad = True)

# a와 b로부터 새로운 텐서를 만든다.
Q = 3*a**3 - b ** 2
"""
    Q에 대해서 .backward()를 호출할 때, autograd는 이러한 gradient를 계산하고 
    이를 텐서의 .grad속성에 저장한다.
    Q는 벡터이므로 Q.backward()에 gradient인자를 명시적으로 전달해야한다.
    gradient는 Q와 같은 모양의 텐서로 Q 자기자신에 대한 변화도를 나타낸다.
    즉 dQ/dQ = 1
    동일하게 Q.sum().backward()와 같이 Q를 스칼라 값으로 aggregate한 뒤 암시적으로
    backward()를 호출할 수도 있다.
"""
external_grad = torch.tensor([1., 1.])
Q.backward(gradient = external_grad)
# 변화도는 a.grad와 b.grad에 저장된다.
print(9*a**2 == a.grad)
print(-2*b == b.grad)

# 선택적으로 읽기 ( Optional Reading ) - autograd를 사용한 벡터 미적분
"""
    수학적으로 벡터함수에서 x에 대한 y의 변화도는 야코비안 행렬이다.
    일반적으로 torch.autograd는 벡터-야코비안 곱을 계산하는 엔진이다.
    이는 주어진 어떤 벡터 v에 대해 J.T * v를 연산한다.
    만약 v가 스칼라 함수 l = g(y)의 변화도인 경우
        v = (dl/dy1, dl/dy2, ..., dl/dym).T
    이며, chain-rule에 따라 벡터 야코비안 곱은 x에 대한 l의 변화도가 된다.
    위 예제에서는 벡터 - 야코비안 곱의 특성을 이용하였으며 external_grad가 v를 뜻한다.
"""

# Computational graph
"""
    개념적으로 autograd는 데이터, 턴세의 실행된 모든 연산들의 기록을 function 객체로
    구성된 DAG에 저장한다. 
"""

# DAG에서 제외하기
"""
    torch.autograd는 requires_grad플래그가 True로 설정된 모든 텐서에 대한 연산들을
    추적한다. 따라서 변화도가 필요하지 않은 텐서들에 대해서는 속성을 False로 설정하여
    DAG 변화도 계산에서 제외합니다.
    입력 텐서 중 단 하나라도 requires_grad= True를 갖는 경우 연산의 결과 텐서도
    grad를 갖게 된다.
"""
x = torch.rand(5,5)
y = torch.rand(5,5)
z = torch.rand((5,5), requires_grad = True)
a = x + y
print(f"Does 'a' require gradients? : {a.requires_grad}")
b = x + z
print(f"Does 'b' require gradients? : {b.requires_grad}")

"""
    신경망에서 변화도를 계산하지 않는 매개변수를 일반적으로 frozen parameter라 부른다.
    이러한 매개변수의 변화도가 필요하지 않는다는 것을 알고있으면 신경망 모델의
    일부를 "freeze (고정)"하는 것이 유리하다. ( 연산량, 메모리 감소 )
    usecase : fine-tuning
    미세조정을 하는 과정에서 새로운 정답을 예측할 수 있도록 모델의 대부분을 고정한 뒤
    일반적으로 classifier layer만 변경한다. 이를 설명하기 위해 예를 들어본다.
"""
for param in model.parameters():
    param.requires_grad = False

# 10개의 정답을 갖는 새로운 데이터셋으로 모델을 미세조정하는 상황을 가정한다.
# resnet 분류기(classifier)는 마지막 선형 계층인 model.fc입니다.
# 이를 새로운 분류기로 동작할 새로운 선형 계층으로 간단히 대체한다.
model.fc = nn.Linear(512,10)
# model.fc를 제외한 모델의 모든 매개변수들이 고정되었으며, 변화도를 계산하는 유일한
# 매개변수는 model.fc의 가중치와 bias뿐이다.

