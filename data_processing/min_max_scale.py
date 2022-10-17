from sklearn.preprocessing import MinMaxScaler
import numpy as np

"""
数据在水平尺度上差别比较大，例如量纲和量级上有差异，原始数据如果直接用，低指标会被削弱到影响忽略不计，高指标会被放大，所以需要进行min_max标准化（归一化）
例如：10，100，10⁴ 这种分布，在广告或者渠道投放场景下使用，例如用户一天内点击1次，点击100次广告投放，投放端还是希望以'一次'的基准计费；
min_max正则化
是将原始数据映射到[0,1]区间
f(xᵢ)=xᵢ/(\sum_{i=1}^{n}x_i)
"""
x_ori = np.array([[1, -1, 3], [3, 0, 1], [0, 1, -1]])
min_max_scale = MinMaxScaler()
x_train_minmax = min_max_scale.fit_transform(x_ori)
print(x_train_minmax)
x_sec = np.array([[-3, -1, 4]])
x_sec_minmax_scala = min_max_scale.transform(x_sec)
print(x_sec_minmax_scala)