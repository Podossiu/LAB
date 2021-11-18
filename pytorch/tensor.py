import torch
import numpy as np

# 텐서 초기화
data = [[1,2],[3,4]]
x_data = torch.tensor(data) # 데이터로부터 텐서 생성

np_array = np.array(data)
x_np = torch.from_numpy(np_array) # numpy로부터 생성

x_ones = torch.ones_like(x_data) # 다른 텐서로부터 생성하기(x_data의 속성 유지 )
print(f"Ones Tensor : \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype = torch.float)
print(f"Random Tensor: \n {x_rand} \n")

# 무작위 값 또는 상수 값을 사용하기
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor : \n {rand_tensor} \n")
print(f"Ones Tensor : \n {ones_tensor} \n")
print(f"Zeros Tensor : \n {zeros_tensor}\n")

# 텐서의 속성: 1. shape, 2. datatype, 3. Device
tensor = torch.rand(3,4)

print(f"Shape of tensor : {tensor.shape}")
print(f"Datatype of tensor : {tensor.dtype}")
print(f"Device tensor is stored on : {tensor.device}")

# 텐서의 연산 ( operation ) gpu 이용
if torch.cuda.is_available():
    tensor = tensor.to('cuda')

# Numpy식의 표준 인덱싱과 슬라이싱
tensor = torch.ones(4,4)
print('First row: ', tensor[0])
print('First column: ', tensor[:,0])
print('Last column: ', tensor[...,-1])

tensor[:,1] = 0
print(tensor)
print(tensor.device)

#텐서 합치기 기준 : tensor dim 1 
t1 = torch.cat([tensor, tensor, tensor], dim = 1) 
print(t1)
print(t1.shape)

# 산술 연산
# matrix multiplication
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)
y3 = torch.rand_like(tensor)

torch.matmul(tensor, tensor.T, out=y3)

# element-wise product
z1 = tensor * tensor
z2 = tensor.mul(tensor)
z3 = torch.rand_like(tensor)
torch.mul(tensor, tensor, out = z3)

print(y1,y2,y3,z1,z2,z3)

# single-element 텐서 ( agg )
agg = tensor.sum()
agg_item = agg.item()
print(agg_item, type(agg_item))

# inplace 연산 ( 바꿔치기) 접미사_
print(tensor, "\n")
tensor.add_(5)
print(tensor)

# Numpy 변환 ( cpu 상의 텐서와 numpy는 메모리 공간을 공유 )
t = torch.ones(5)
print(f"t : {t}")
n = t.numpy()
print(f"n : {n}")

t.add_(1)
print(f"t : {t}")
print(f"n : {n}")

# Numpy 배열을 텐서로 변환하기
n = np.ones(5)
t = torch.from_numpy(n)

np.add(n, 1, out = n)
print(f"t: {t}")
print(f"n: {n}")

