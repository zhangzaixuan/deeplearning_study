import numpy as np
from sklearn import datasets

"""
01 其实pytorch 在工程层面分为4个大模块，其他场景不太熟悉，感觉好多算法的代码好零散，应该有企业大拿回去教书或者写些好的算法工程教程；
02 推荐和nlp要处理的5步应该够了，数据/模型/损失函数/优化器/迭代训练 结果是模型，最重要的反倒是数据或者说特征，训练数据决定了模型效果上限，模型感觉更有点像夹逼准则一样，可能我的数学底子还是太弱了，当初应该好好学下数理逻辑和小波分析的；
03 一直没想过商业化模型或者被下发烧录的算法怎么加密，逆向工程始终有办法能解密，或者封装成一个本地服务或者宏，嵌在go或者c++的代码中，对c++,go的代码反编译来做整体加密？或者在tvm里面搞掂东东；
"""
"""
sklearn的数据集接口分为3种：load,fetch,make
load 主要是引用sklearn内置数据集合，学习的时候用的比较多；
fetch 一般都通过在线链接下载数据，可以先在网站上下载数据集，然后自己装配；
make 自己本地加工生成的数据集；
生产各家公司都不一样，以前公司主要从hive和hbase读数据，加工成dataframe(spark),用arrow加速，数据集如果不大的话，pandas读数据库也可以，风控的有些模型;
"""
# 获取内置数据集，sklearn.datasets.load...,在线下载数据集，sklearn.datasets.fetch,
# svmlight/libsvm格式数据集， sklearn.datasets.load_svmlight_file()
# mldata.org sklearn.datasets.fetch_mldata()/fetch_openml() 旧版接口/新版
iris_data = datasets.load_iris()
iris_data_slice = iris_data['data'][:5]
print(iris_data_slice)
# 数据切片
"""
iris_data_slice
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
"""
iris_target = np.unique(iris_data['target'])
# [0 1 2]
print(iris_target)
# feature_names 参数名称 花瓣的一些长度和宽度
iris_feature_names = iris_data['feature_names']
print(iris_feature_names)
# ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# target_names 品类名称
iris_target_names = iris_data['target_names']
print(iris_target_names)
# ['setosa' 'versicolor' 'virginica']

# 这个类似于编程语言的helloworld,下不到数据集估计得科学上网，买个网络vpn
mnist = datasets.fetch_openml('mnist_784', version=1)

# 创建数据集方法,make_blobs 聚类和分类,make_classification 聚类,make_circles 聚类,make_moons 聚类,make_regression 回归

from sklearn.datasets._samples_generator import make_blobs, make_classification, make_circles, make_moons, \
    make_regression
# 需要的参数如下，追进去看一下就可以，基本都是信息熵里面的一些参数概念
# n_samples 样本数,centers 聚类中心，n_features 特征数 cluster_std 聚类标准差
make_blobs()
# n_classes 分类标签数 n_redundant 多余特征数
make_classification()
# noise 高斯噪声标差 factor 内外圆的比例因子 0～1
make_circles()
make_moons()
# bias 需指定偏差
make_regression()
