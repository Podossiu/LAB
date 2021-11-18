# Torch.Autograd를 사용한 자동 미분
# 신경망을 학습할 때 torch.autograd라고 불리는 자동 미분 엔진이 도입되어있음
# 모든 computational graph에 대한 gradient의 값을 자동계산해줌
import torch
x = torch.ones(5) # input tensor
y = torch.zeros(3) # expected output
w = torch.randn(5, 3, requires_grad = True)
b = torch.randn(3, requires_grad = True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
# requires_grad의 값은 텐서를 생성할 때 설정하거나 나중에 x.requires_grad_(True) 메소드를 사용하여 나중에 설정할 수 있다.

# 연산 그래프를 구성하기 위해 텐서에 적용하는 함수는 Function class의 객체이다. 이 객체는 순전파 방향으로 함수를 계산하는 방법과 역전파 단계에서 도함수를 계산하는 방법을 알고 있다. 

# 역방향 전파 함수에 대한 참조 ( reference )는 텐서의 grad_fn 속성에 저장된다.

print('Gradient function for z = ', z.grad_fn)
print('Gradient function for loss =', loss.grad_fn)

# 변화도 계산하기
# 신경망에서 매개변수의 가중치를 최적화하려면 손실함수의 도함수를 계산해야 한다. 즉, x와 y의 일부 고정값에서 dloss/dw와dloss/db가 필요하다. 이러한 도함수를 계산하기 위해 loss.backward()를 호출 한 뒤 w.grad와 b.grad에서 값을 가져온다.
loss.backward()
print(w.grad)
print(b.grad)

# 연산 그래프의 leaf 노드중 requires_grad속성이 True로 설정된 노드들의 grad속성만 구할 수 있다. 그래프의 다른 모든 노드에서는 변화도가 유효하지 않다.
# 성능상의 이유로 주어진 그래프에서 backward를 사용한 변화도 계산은 한 번만 수행할 수 있다. 만약 동일한 그래프에서 여런번 backward호출이 필요한 경우에는 backward호출 시 retain_graph = True를 전달해야 한다.

# 변화도 추적 멈추기
# 기본적으로 requires_grad=True인 모든 텐서들은 연산 기록을 추적하고 변화도 계산을 지원한다. 그러나 모델을 학습한 뒤 입력 데이터를 단순히 적용하기만 하는 경우 ( 순전파 연산만 필요 ) 이러한 추적이나 지원이 필요하지 않을 수 있따. 연산코드를 torch.no_grad() 블록으로 둘러싸서 연산 추적을 멈출 수 있다.

z = torch.matmul(x,w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x,w) + b
print(z.requires_grad)

# 동일한 결과를 얻는 방법은 텐서에 detach() 메소드를 사용하는 것이다.
z = torch.matmul(x,w) + b
z_det = z.detach()
print(z_det.requires_grad)

# 변화도 추적을 멈춰야 하는 이유
# 신경망의 일부 매개변수를 고정된 매개변수 ( frozen parameter )로 표시한다. 사전 학습된 신경망을 미세조정 할 때 매우 일반적인 시나리오이다.
# 변화도를 추적하지 않는 텐서의 연산이 더 효율적이기 때문에, 순전판 단계만 수행할 때 연산속도가 향상된다.

# 연산 그래프에 대한 추가 정보
# 개념적으로 autograd는 데이터(텐서)의 실행된 모든 연산들 ( 및 연산 결과가 텐서인 경우도 포함 ) 의 기록을 Function객체로 구성된 방향성 비순환 그래프 ( DAG ) 에 저장한다. 이 DAG의 leaf는 입력 텐서이며, root는 결과 텐서이다. 이 그래프를 root부터 leaf까지 추적하여 chain-rule에 의해 변화도를 자동으로 계산한다.

"""
    순전파 단계에서 autograd는 다음 두가지 작업을 동시에 수행한다.
    - 요청된 연산을 수행하여 결과 텐서를 계산한다.
    - DAG에 연산의 변화도 기능 ( gradient function )을 유지한다.

    역전파 단계는 DAG 뿌리에서 .backward()가 호출될 때 시작된다. autograd는 이 때
    - 각 grad_fn으로부터 변화도를 계산한다.
    - 각 텐서의 .grad 속성에 계산 결과를 accumulate한다.
    - 연쇄 법칙을 사용하여 모든 leaf 텐서들까지 전파한다.

    Pytorch에서 DAG는 동적이다. 주목해야 할 중요한 점은 그래프가 from scratch부터 다시 
    생성된다는 것이다. 매번 .backward()가 호출되고 나면, autograd는 새로운 그래프를
    채우기 시작한다. 이러한 점 덕분에 모델에서 흐름제어 ( control flow )구문을 사용할
    수 있는 것이다. 매번 iterate마다 필요하면 모양 ( shape )나 크기 ( size ), 연산
    ( operation )을 바꿀 수 있다.
"""

# 선택적으로 읽기 ( optional Reading ) : 텐서 변화도와 야코비안 곱 
"""
    대부분의 경우 스칼라 손실 함수를 가지고 매개변수와 관련된 변화도를 계산해야한다.
    그러나 출력함수가 임의의 텐서인 경우가 있다. 이럴 때 Pytorch는 실제 변화도가 아닌
    야코비안 곱을 계산한다.
    야코비안 행렬 자체를 계산하는 대신, Pytorch는 주어진 입력 벡터에 대한 야코비안 
    곱을 계산한다. 이 과정은 v를 인자로 backward를 호출하면 이뤄지며 v의 크기는 
    곱을 계산하려고 하는 원래 텐서의 크기와 같아야 한다.
"""

inp = torch.eye(5, requires_grad = True)
out = (inp+1).pow(2)
out.backward(torch.ones_like(inp), retain_graph = True)
print("First call\n", inp.grad)

out.backward(torch.ones_like(inp), retain_graph = True)
print("Second call\n", inp.grad)

inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph = True)
print("\nCall after zeroing gradients\n", inp.grad)




