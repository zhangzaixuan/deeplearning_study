from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris

iris = load_iris()
# embedded 嵌入法本质上是一种过滤，过滤过程基于学习模型训练，获取各个特征的权值系数，根据权值系数从大到小筛选特征，常用惩罚项和树模型嵌入
# Solver lbfgs supports only 'l2' or 'none' penalties, got l1 penalty.
L1_model = SelectFromModel(LogisticRegression(C=0.1)).fit_transform(iris.data, iris.target)[:5]
L2_model = SelectFromModel(LogisticRegression(penalty="l2", C=0.1)).fit_transform(iris.data, iris.target)[:5]
"""
基于惩罚项嵌入来说，正则化的惩罚项越大，权重系数越小，正则化主要为L1和L2正则
"""
print("l1 逻辑回归：" + str(L1_model) + '\n' + "l2 逻辑回归：" + str(L2_model))

# 基于树模型的嵌入，一般需要决策树和随机森林的组合
decision_tree_model = SelectFromModel(DecisionTreeRegressor()).fit_transform(iris.data, iris.target)[:5]
random_forest_model = SelectFromModel(RandomForestRegressor()).fit_transform(iris.data, iris.target)[:5]

print("决策树：" + str(L1_model) + '\n' + "随机森林：" + str(L2_model))

