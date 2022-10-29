from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
"""
wrapper包装，通过预测模型效果评分，每次剔除质量不太好的特征，保留权值系数和目标或者结果关联性比较好的特征，
其实就是将权重系数比较低的特征移除，因为移除这部分特征，预测和实际的模型效果本身的拟合度偏离很小
选择比较有代表性的特征消除法就是RFE（递归特征消除），例如总特征数为n,基于基模型，第一轮rfe选择目标特征数为k(k<n)，第一轮特征筛选
完成后，第二轮选择为i(i<k),第三轮选择为j(j<i),特征量级不大执行等差数列减少，每轮-1，-2，如果特征数据过多，可以以10
，20 这种方式执行正交特征消除，每一轮次的模型更新作为下一轮次的新模型；
tips:模型相对其实还是比较好处理的，生产比较难以规避的其实是模型遗忘和模型过拟合(特殊场景下危害比较大，所以尽量在后向做部分规则处理和人工纠偏)
"""
iris = load_iris()
# estimator 基础模型 LogisticRegression 逻辑回归,和resnet50 一样属于万金油基础模型，n_features_to_select 目标保留特征数
rfe_res = RFE(estimator=LogisticRegression(), n_features_to_select=3).fit_transform(iris.data, iris.target)[:5]
print(rfe_res)
