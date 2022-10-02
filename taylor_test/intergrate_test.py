import sympy
import math
import numpy as np

# x = sympy.symbols("x")
"""
x**2 积分函数
x 积分变量
1,2 积分区间
"""
# print(sympy.integrate(x ** 2, (x, 1, 2)))
# x = sympy.symbols('x', positive=True)
# r = sympy.symbols('r', positive=True)

x = sympy.symbols('x')
r = sympy.symbols('r', positive=True)

# def func(r, x):
#     return math.sqrt(r ** 2 - x ** 2)
# circle_area = 2 * sympy.integrate(func(r, x), (x, -r, r))

print(3 ** 2)

circle_area = 2 * sympy.integrate(sympy.sqrt(r ** 2 - x ** 2), (x, -r, r))

# circle_area = 2 * sympy.integrate(np.sqrt(r ** 2 - x ** 2), (x, -r, r))

print(circle_area)
print(sympy.integrate(circle_area, (x, -r, r)))

# print(math.sqrt(2.0))
# print(math.sqrt(2))

"""
顶你个肺啊，sympy和numpy,math python3 中不要混用，会报convert error
"""
