from scipy import stats
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong']
plt.rcParams['axes.unicode_minus'] = False

"""
二项分布为n重独立伯努利分布，每次的伯努利彼此独立
伯努力实验可以看作n=1时的二项分布；

随机变量X服从参数为n,p的二项分布，记作X～B（n,p）
P{X=k}=Cᵏₙ*pᴷ*(1-p)ⁿ-ᵏ Cᵏₙ=n!/k!(n-k)!"""
for prob in range(3, 10, 3):
    x = np.arange(0, 25)
    binom = stats.binom.pmf(x, 20, 0.1 * prob)
    plt.plot(x, binom, '-o', label="p={:f}".format(0.1 * prob))
    plt.xlabel('Random Variable 随机值', fontsize=12)
    plt.ylabel('Probability 概率值', fontsize=12)
    plt.title("binomial distribution 二项式分布")
    plt.legend()

plt.savefig('./distribution_img/binomial_distribution.png')
plt.pause(1)
