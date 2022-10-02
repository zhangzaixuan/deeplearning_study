from scipy import integrate

# 单纯一重积分请参看taylor_test,这里为二重三重积分的定积分例子
# 二重积分使用dblquad方法
"""
lambda y, x dx,dy的意思，为x,y的微分变量
x * y ** 2 积分函数
"""
f = lambda y, x: x * y ** 2
"""
f,积分函数
0，自变量x下界，左区间
2，自变量x上界，右区间
lambda x: 0 因变量y下界0
lambda x: 1 因变量y上界1
以当前为例,0与2叫作x积分下限与积分上限，区间[0,2]叫作积分区间，函数f(x,y)=x * y ** 2叫作被积函数，x,y叫作积分变量，f(x)dxdy叫作被积式
"""
# 这个返回很像go的异常处理,但是不是，这里err1为abserr,为积分计算误差
val1, err1 = integrate.dblquad(f, 0, 2, lambda x: 0, lambda x: 1)
print(err1)
print('double integral result:', val1)

# 三重积分使用tplquad方法
"""
lambda z, y, x: x * y * z lambda z,y,x dxdydz ,f(x,y,z)=x*y*z 积分表达式 xyzdxdydz
0, 3 x 0->3
lambda x: 0, lambda x: 2, y 0->2
lambda x, y: 0,lambda x, y: 1 z 0->3
"""
val2, err2 = integrate.tplquad(lambda z, y, x: x * y * z, 0, 3, lambda x: 0, lambda x: 2, lambda x, y: 0,
                               lambda x, y: 1)
print('three integral result:', val2)
