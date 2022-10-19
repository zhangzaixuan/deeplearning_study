import numpy as np
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
X_ORI = [[7, 2, 3], [4, np.nan, 6], [10, 5, 9]]
imp_mean.fit(X_ORI)
print(X_ORI, '\n', imp_mean.statistics_)
# 重新实例化
SimpleImputer()
X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]
print(X, '\n', imp_mean.transform(X))
"""
missing_values 填充应该匹配哪些异常值；
strategy 填充策略；mean 均值填充，均值计算逻辑看按列还是行取均值，median：中位数，most_frequent: 众数
默认按照行进行列字段的缺失值填充,应该就是以前的axis=0,现在api里面移除这个参数选项了
[[7, 2, 3], [4, nan, 6], [10, 5, 9]] 
 [7.  3.5 6. ]
[[nan, 2, 3], [4, nan, 6], [10, nan, 9]] 
 [[ 7.   2.   3. ]
 [ 4.   3.5  6. ]
 [10.   3.5  9. ]]
"""