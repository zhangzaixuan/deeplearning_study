from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
from scipy.stats import pearsonr
import pandas as pd
import numpy as np

iris = load_iris()
iris_df = pd.DataFrame(iris.data[:5], columns=iris['feature_names'])
print(iris_df)
# 特征选择方式（还是那句经典的话，数据特征决定学习结果的上限，模型和网络结果只是尽可能去逼近这个上限）
# 001 方差选择法，如果某一列数据的方差过小，说明数据分布在这个特征列上其实没太大差异，这一列数据从整体数据特征上剔除对数据的分布不会产生太大影响；
iris_var = VarianceThreshold(threshold=3).fit_transform(iris.data)
print(pd.DataFrame(iris_var))
"""
   特征分布列方差大于3的特征列为petal length (cm)
   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
0                5.1               3.5                1.4               0.2
1                4.9               3.0                1.4               0.2
2                4.7               3.2                1.3               0.2
3                4.6               3.1                1.5               0.2
4                5.0               3.6                1.4               0.2
       0
0    1.4
1    1.4
2    1.3
3    1.5
4    1.4
"""
# threshold 方差阈值，小于这个方差阈值的数据特征被丢弃，不填充的默认值为：threshold=0.0
iris_var1 = VarianceThreshold().fit_transform(iris.data)
print(iris_var1)

# 002 相关系数
"""
相关系数计算公式 r=\sum_{i=1}^{n}(x_i-\overline{x})(y_i-\overline{y})/\sqrt{\sum_{i=1}^{n}(x_i-\overline{x})²}\sqrt{\sum_{i=1}^{n}(y_i-\overline{y})²}
相关系数反应的是特征对于目标值的相关性，系数越大，变量之间的关系越强
"""
# 函数将返回一个二元组数组，数组第i个值 （第i个特征的评分和P值）
r_function = lambda X, Y: np.array(list(map(lambda x: pearsonr(x, Y)[0], X.T))).T
# r_function 评估特征和目标关联性的函数，k 为选择相关性好的特征数量；
features_k = SelectKBest(r_function, k=2).fit_transform(iris.data, iris.target)[:5]
print(features_k)
"""
输出结果是花瓣长度和花瓣宽度：petal length (cm)  petal width (cm)
[[1.4 0.2]
 [1.4 0.2]
 [1.3 0.2]
 [1.5 0.2]
 [1.4 0.2]]
"""

# 003 卡方校验
"""
卡方校验是判定样本分布是否符合特定分布的校验方法，检验观察值和理论值的吻合度（皮尔逊系数，数分大佬口里念叨的皮尔森，随意吧）
卡方计算方式 \sum_{i=1}^{n}(A-T)²/T A为实值，T为理论值
"""
chi2_k = SelectKBest(chi2, k=2).fit_transform(iris.data, iris.target)[:5]
print(chi2_k)
# 筛选结果是花瓣长度和花瓣宽度,chi2 为卡方函数，k为需筛选特征数量
