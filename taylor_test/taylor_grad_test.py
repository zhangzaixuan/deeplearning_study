import numpy as np
import math
import matplotlib as mpl
import matplotlib.pyplot as plt


def calc_e_small(x):
    n = 10
    f = np.arange(1, n + 1).cumprod()
    b = np.array([x] * n).cumprod()
    return np.sum(b / f) + 1


def calc_e(x):
    reverse = False
    if x < 0:
        x = -x
        reverse = True
    y = calc_e_small(x)
    if reverse:
        return 1 / y
    return y


if __name__ == '__main__':
    t1 = np.linspace(-2, 0, 10, endpoint=False)
    t2 = np.linspace(0, 4, 20)
    t = np.concatenate((t1, t2))
    y = np.empty_like(t)
    for i, x in enumerate(t):
        y[i] = calc_e(x)
        print('e^', x, '=', 'y[i]', '近似值\t', math.exp(x), '(真实值)')

    """
    matplotlib版本过高报错：
    module 'backend_interagg' has no attribute 'FigureCanvas'
    需降低plt版本
    pip uninstall matplotlib
    pip install matplotlib==3.5.3
    """
    plt.figure(facecolor='w')
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.plot(t, y, 'r-', t, y, 'go', linewidth=2, markeredgecolor='k')
    plt.title('taylor-expansion', fontsize=18)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('exp(X)', fontsize=15)
    plt.grid(True, ls=':')
    # plt.show()
    plt.draw()
    epoch = 0
    plt.savefig('./out_data/image/pic-{}.png'.format(epoch + 1))
    plt.pause(1)
    # plt.close()
