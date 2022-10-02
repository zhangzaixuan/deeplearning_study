import sympy

"""
单导数example
"""
x = sympy.Symbol("x")
print(sympy.diff(x ** 3, x))
print((x ** 2).diff())

"""
偏导数example
"""
x1 = sympy.Symbol("x1")
x2 = sympy.Symbol("x2")
y = x1 ** 3 + x1 * x2 + x2
print(sympy.diff(y, x1))
print(sympy.diff(y, x2))

"""
表达式带入求导
"""


def sympy_derivative():
    x = sympy.Symbol("x")
    y = x ** 3 + x
    return sympy.diff(y, x)


func = sympy_derivative()
print(func)
print(func.evalf(subs={'x': 6}))

"""
偏导数代值计算同上
"""