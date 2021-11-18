# numpy는 훌륭한 framework지만 GPU를 사용하여 가속화 할 수 없음 
"""
    Tensor는 개념적으로 Numpy배열과 동일하다. Tensor는 n차원 배열이며,
    Pytorch는 이러한 tensor들의 연산을 위한 다양한 기능들을 제공한다.
    tensor를 통해서 삼차 다항식을 sine함수에 근사한다.
"""
import torch
import math

dtype = torch.float
device = torch.device("cpu")

x = torch.linspace(-math.pi, math.pi, 2000, device = device, dtype = dtype)
y = torch.sin(x)

# 무작위로 가중치 초기화
a = torch.randn((), device = device, dtype = dtype )
b = torch.randn((), device = device, dtype = dtype )
c = torch.randn((), device = device, dtype = dtype )
d = torch.randn((), device = device, dtype = dtype )

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계 : 예측값 y를 계산한다.
    y_pred = a + b * x + c * x ** 2 + d * x ** 3

    # loss를 계산하고 출력한다.
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss) # 100 epoch마다 loss 출력
    # 손실에 따른 a, b, c, d의 gradient를 계산하고 역전파
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = (grad_y_pred).sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x ** 2).sum()
    grad_d = (grad_y_pred * x ** 3).sum()

    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3')

