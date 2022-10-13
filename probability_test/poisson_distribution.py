from scipy.stats import binom
import matplotlib.pyplot as plt
import numpy as np

# 泊松分布 单位时间单位面积内随机事件发生概率 期望和方差都为lamda
"""
P(x=i)=λ^i/λ!(e^-λ)(i=1,2....,n)
"""
plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False
# lam lamda size i
x = np.random.poisson(lam=5, size=10000)
s = plt.hist(x, bins=15, range=[0, 15], color='g', alpha=0.5)
plt.plot(s[1][0:15], s[0], 'r')
plt.grid()
# plt.show()
plt.savefig('./distribution_img/poisson.png')
plt.pause(1)
