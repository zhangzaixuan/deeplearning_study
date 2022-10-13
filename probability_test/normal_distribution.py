import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False

"""
正态分布，这个应该是传播度最高的概率分布函数了，这里不做解释了，基本没人没听过六西格玛原则
"""
mean = 0
n = np.arange(-100, 100)
normal = stats.norm.pdf(n, mean, 20)
plt.plot(n, normal)
plt.xlabel('Distribution', fontsize=12)
plt.xlabel('Probability', fontsize=12)
plt.title("Normal Distribution 正态分布")
plt.savefig('./distribution_img/normal_distribution.png')
plt.pause(1)
