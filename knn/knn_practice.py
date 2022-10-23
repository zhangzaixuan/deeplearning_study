import numpy as np
from sklearn import datasets
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

"""
knn,全称 K-NearestNeighbor（近邻算法）分类算法（分类和回归其实没有明显的分界，主要看解决问题所需要的输入，
分类可以看作特殊的回归，回归设定一个数据分阈，就可以转换为分类）

knn的工程定义，每个样本可以通过距离最接近（看定义）的k个邻居来表征，这种基于距离聚合的样本可以共用相同的标签，说人话就是 '人以类聚，物以群分'
举个例子，在社交场景，用户可以通过登陆次数，发帖数，发帖内容分类，浏览帖子内容分类，pugc数，会话回合数，派对次数，上麦次数，会员等级，消费钻石等虚拟资产数
给用户构建标签 例如社交达人，绝壁之花，高岭之花，土豪玩家，文艺青年，氛围能手，后续可以基于这些标签进行内容召回和用户推送。

距离的定义可以执行向量计算，也可以根据用户行为或者用户积分等进行score分值之间的交叉组合划分，无需进行相似度计算；

例如交互分75-80，活跃分90-100，帖子浏览分40-60，消费等级分70-80 就可以定义 氛围小能手标签，一个用户可以被打上多个标签，
标签：（冷热，冷标签 长时标签，热标签，短时标签，用户行为时间比较近）
      标签可以具有相似性，也可以具有互斥性

标签使用：
  场景1：基于用户行为事件，输入事件规则引擎，实时计算或者近实时计算热标签，匹配冷标签，进行用户在线推荐，feed 瀑布流；
  场景2：作为uer-cf矩阵的一个或者多个特征值,进行千人千面的推荐场景(deepfm,als...);
  场景3：用于用户运营的用户圈选（和mpp数据库,bitmap数据结构相结合）;
  
注意点：01 包括knn在内的多种算法单一效果并不最佳，后续需要添加多路match和多级rank(粗排和精排)，然后混合全域热度和分类top的内容才可以推送
       02 场景不一样，推荐的方式不同，取决于内部或者客户的目标，广告投放考虑的曝光度和适配度，社交产品考虑的社区和内容运营和当时充值消费，电商场景需要考量的是gmv(长时会更难)
       
某种程度上，我们是自由意志的罪人,希望以后会有好的解决方案。
"""
iris = datasets.load_iris()
X = iris.data
y = iris.target
X_train, X_Test, y_train, y_test = train_test_split(X, y, random_state=0)


def get_dis(ins1, ins2):
    """
    获取两个数据之间的欧氏距离
    :param ins1: 样本1,数据类型 array
    :param ins2: 样本2
    :return: 欧氏距离计算值
    """
    dist = np.sqrt(sum((ins1 - ins2) ** 2))
    return dist


k_num = 3


def knn_classify_iter(X, y, data_test, k):
    """
    使用knn获取测试数据集的标签
    :param X: 训练数据特征
    :param y: 训练数据标签
    :param data_test: 测试数据集
    :param k: 结果划分的类别数目
    :return: 获取标签权值最高的（排序最高的）
    """
    # 获取预测标签
    x_dis = [get_dis(i, data_test) for i in X]
    # 对标签排序
    k_neighbors = np.argsort(x_dis)[:k]
    # 获取排序标签计数
    neighbor_count = Counter(y[k_neighbors])
    # 获取排序最大的 Find  the largest elements in a dataset
    return neighbor_count.most_common()[0][0]


# 这里预知了一下，iris是经典的三分类数据集，所以指定k_num为3，正常应该给定范围循环测试或者使用grid检索最佳k_num
iris_predictions = [knn_classify_iter(X_train, y_train, i, k_num) for i in X_Test]
print(iris_predictions[:5])
predict_matchs = np.count_nonzero((iris_predictions == y_test) == True)
clf = KNeighborsClassifier(n_neighbors=k_num)
clf.fit(X_train, y_train)
print('knn model accuracy is :%.5f' % (accuracy_score(y_test, clf.predict(X_Test))))
print('current knn model accurayc is :%.5f' % (predict_matchs / len(X_Test)))
