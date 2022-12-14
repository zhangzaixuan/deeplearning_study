import numpy as np

# 行列式概念相对矩阵而言，行列式是矩阵导出的一个具有未知数的表达式,A nxn det(A),|A|为A的行列式
"""
二阶行列式的计算为主对角线乘积-副对角线乘积
超过二阶行列式的求解使用代数余子式和化三角形法

代数余子式 将元素所在的行和列剔除，单一元素和剔除后行和列的行列式（n-1）x(n-1)相乘,然后累加，单一乘积累加的正负系数规则为
Aij=(-1)i+jMij 此处i+j为指数

化三角形法，通过矩阵变换获取结果行列式，最终目标，将行列式左下方全部化为0，行列式计算依赖性质
01 对换行列式中两列（行）的位置，行列式反号
02 将行列式的某一行列的倍数加到另一行（列），行列式值不变 
其实可以理解，上述的两个性质正向和逆向操作是幂等的
"""
A = np.mat([[1, 0, 3], [2, 0, 1], [3, 3, 5]])
print(np.linalg.det(A))
A1 = np.mat([[1, -1, 2, -3, 1], [-3, 3, -7, 9, -5],
             [2, 0, 4, -2, 1], [3, -5, 7, -14, 6], [4, -4, 10, -10, 2]])
print(A1)
print(np.linalg.det(A1))
