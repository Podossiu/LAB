# 오래된 파이토치 구현
"""
    numpy를 통해서 신경망을 구현한다. 3차 다항식이 sine함수에 근사하도록 
    순전파 단계와 역전파 단계를 직접 구현해본다.
"""
import numpy as np
import math

# 입력과 출력 데이터 생성
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

# 무작위로 가중치를 초기화함
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()

learning_rate = 1e-6
for t in range(2000):
    # 순전파 단계 y 값을 예측한다.
    # y = a + bx + c x^2 + d x^3
    y_pred = a + b * x + c * x ** 2 + d * x ** 3
    
    # 손실을 계산하고 출력함 SE
    loss = np.square(y_pred - y).sum()
    # 100개마다 손실 출력
    if t % 100 == 99:
        print(t, loss)
    #손실에 따른 a,b,c,d의 변화도를 계산하고 역전파한다.
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = ( grad_y_pred * x ).sum()
    grad_c = ( grad_y_pred * x ** 2 ).sum()
    grad_d = ( grad_y_pred * x ** 3 ).sum()

    # 가중치를 갱신한다.
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f'Result: y = {a} + {b} x + {c} x^2 + {d} x^3')

