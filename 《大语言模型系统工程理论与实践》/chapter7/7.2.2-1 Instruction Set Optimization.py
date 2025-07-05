# CPU上使用AVX-512指令集加速矩阵乘法示例
import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

def optimized_matmul(A, B):
    # sklearn的safe_sparse_dot会自动利用AVX指令集
    return safe_sparse_dot(A, B)

# 与普通numpy实现对比
def baseline_matmul(A, B):
    return np.matmul(A, B)

# 性能对比
import time
A = np.random.random((1024, 1024)).astype(np.float32)
B = np.random.random((1024, 1024)).astype(np.float32)

start = time.time()
C1 = optimized_matmul(A, B)
end = time.time()
print(f"Optimized matmul: {end - start:.4f}s")

start = time.time()
C2 = baseline_matmul(A, B)
end = time.time()
print(f"Baseline matmul: {end - start:.4f}s")
