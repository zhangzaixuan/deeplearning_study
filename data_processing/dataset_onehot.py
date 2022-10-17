import pandas as pd
from sklearn.preprocessing import OneHotEncoder

"""
onehot 独热编码将分类变量转换成向量,独热编码多用于nlp，蛋白质分子结构，化学键之类的编码中
"""
sex = ['male', 'female']
sex_pd_onehot = pd.get_dummies(sex)
print(sex_pd_onehot)
print(type(sex_pd_onehot))
"""
   female  male
0       0     1
1       1     0
<class 'pandas.core.frame.DataFrame'>
"""
# sklearn 的onehot编码数据需要嵌套；
sklearn_sex = [['male'], ['female']]
sk_onehot = OneHotEncoder()
sex_sk_onehot = sk_onehot.fit_transform(sklearn_sex)
print(sex_sk_onehot)
print(sex_sk_onehot.toarray())
# 输出原来的分类变量
print(sk_onehot.categories_)
"""
<class 'pandas.core.frame.DataFrame'>
  (0, 1)	1.0
  (1, 0)	1.0
[[0. 1.]
 [1. 0.]]
[array(['female', 'male'], dtype=object)]
"""
x_mul = [[0, 0, 3],
         [1, 1, 0],
         [0, 2, 1],
         [1, 0, 2]]
mul_one_hot = sk_onehot.fit_transform(x_mul)
print(mul_one_hot, mul_one_hot.toarray(), sk_onehot.categories_)
"""
(0, 0)	1.0
  (0, 2)	1.0
  (0, 8)	1.0
  (1, 1)	1.0
  (1, 3)	1.0
  (1, 5)	1.0
  (2, 0)	1.0
  (2, 4)	1.0
  (2, 6)	1.0
  (3, 1)	1.0
  (3, 2)	1.0
  (3, 7)	1.0 
[[1. 0. 1. 0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 1. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 1. 0. 1. 0. 0.]
 [0. 1. 1. 0. 0. 0. 0. 1. 0.]] 
         [[0, 0, 3],
         [1, 1, 0],
         [0, 2, 1],
         [1, 0, 2]]
 第一列有0，1两个特征，第二列有0，1，2 三个特征，第三列有0，1，2，3 四个特征，一共有9个特征
 [1. 0. 1. 0. 0. 0. 0. 0. 1.]->[0, 0, 3] [1.0]->0,[1.0.0]->0,[0.0.0.1]->3
 
 [array([0, 1]), array([0, 1, 2]), array([0, 1, 2, 3])]

"""
tag = [['a', 'a', 'd'], ['b', 'b', 'a'], ['a', 'c', 'b'], ['b', 'a', 'c']]
tag_one_hot = OneHotEncoder()
print(tag_one_hot.fit_transform(tag).toarray())
print(tag_one_hot.categories_)
"""
[[1. 0. 1. 0. 0. 0. 0. 0. 1.]
 [0. 1. 0. 1. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 1. 0. 1. 0. 0.]
 [0. 1. 1. 0. 0. 0. 0. 1. 0.]]
[array(['a', 'b'], dtype=object), array(['a', 'b', 'c'], dtype=object), array(['a', 'b', 'c', 'd'], dtype=object)]
"""
