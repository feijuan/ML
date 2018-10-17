import numpy as np
from numpy import *
import matplotlib.pyplot as mlt

A = mat([[8, -3, 2], [4, 11, -1], [6, 3, 12]])
b = mat([20, 33, 36])
result = np.linalg.solve(A, b.T)
print(linalg.inv(A).dot(b.T))
B0 = mat([[0, 3/8, -2/8], [-4/11, 0, 1/11], [-6/12, -3/12, 0]])
f = mat([20/8, 33/11, 36/12])
error = 1.0e-6  # 误差阈值
steps = 100
xk = zeros((3, 1))
errorlist = []
for k in range(steps):
    xk_1 = xk
    xk = B0*xk+f.T
    errorlist.append(linalg.norm(xk-xk_1))
    if errorlist[-1] < error:
        print(k+1)
        break
print(xk)
