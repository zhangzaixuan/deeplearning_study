from scipy import stats
import math
import numpy as np
import matplotlib.pyplot as plt

"""
指数分布 两次事件发生的时间间隔，连续随机变量概率分布

概率密度函数：
f(x) 1/θ*(e^(-x/θ)), x>0;0,x<=0
1/0 指数分布参数，每单位时间内发生事件的次数
指数期望 E(x)=0,方差D(x)=0²
指数分布函数F(x) 1-e^(-x/0),x>0;0,x<=0;
"""

plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False
# r = 1 / 50000
r = 0.00002
print(r)
x_list = []
y_list = []
for x in np.linspace(0, 1000000, 100000):
    if x == 0:
        continue
    # p = r * math.e ** (-r * x)
    p = stats.expon.pdf(x, scale=1 / r)
    x_list.append(x)
    y_list.append(p)
plt.plot(x_list, y_list)
plt.xlabel("时间间隔")
plt.ylabel("概率密")
# plt.show()
plt.savefig('./distribution_img/exponential.png')
plt.pause(1)
