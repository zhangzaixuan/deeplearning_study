import numpy as np

# 从数组获取矩阵，矩阵Amxn,m行n列元素
mat1 = np.mat([[1, 2], [3, 4]])
print(mat1)
x = np.array([[1, 2], [3, 4]])
mat_x = np.asmatrix(x)
print(mat_x)
print(x[0, 0])

# 矩阵的线性计算
# 矩阵的相加减需要保证两个矩阵的行数和列数都相等
A = np.array([[1, 2], [3, 4]])
B = np.array([[1, 1], [1, 1]])
print(A + B)
print(A - B)
  
# 矩阵A和B，需要A的列数等于B的行数 C=AxB，Amxn,Bnxs Cmxs cij=ai1b1j+ai2b2j+......+ainbnj

A1 = np.mat([[1, 2, 3], [4, 5, 6]])
B1 = np.mat([[1, 2], [3, 4], [5, 6]])
dot1 = A1 * B1
dot2 = B1 * A1
print(dot1, '\n', dot2)
print(np.dot(A1, B1), '\n', np.dot(B1, A1))
print(A1 @ B1, '\n', B1 @ A1)

# 单位矩阵，nxn 的方阵，主对角线上元素全为1，其他元素为0，主对角线，左上角到右下角，与之相对应的是副对角线，单位矩阵乘任一矩阵等于原矩阵
square_mat = np.mat(np.eye(4))
print(square_mat)