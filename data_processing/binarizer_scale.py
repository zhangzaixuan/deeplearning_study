import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import LabelBinarizer

"""
二值化，将特征和标签进行0，1 二分类，特征可以直接转，标签因为是文本数据，需要一定映射和一定逻辑分类，例如上海，北京，陕西，苏州 陕西，北京，例如归于北方0，苏州，上海归于南方1;
数据标准或者归一化的操作可以的话还是通过数据库结果集或者视图物化列多种方式来操作，在程序中处理不算一个好主意
"""
x_ori = [[1, 1, 2], [2, 0, 0], [0, 1, -1]]
binarizer = Binarizer()
label_binarizer = LabelBinarizer()
# fit_transform 这个不要纠结，伴生apply函数，Fit to data, then transform it
x_bina_trans = binarizer.fit_transform(x_ori)
print(x_bina_trans)

label_list = ['Y', 'N', 'Y', 'N']
# fit_transform,inverse_transform 标签的正向和逆向函数
male_trans = label_binarizer.fit_transform(label_list)
gender_class_inve = label_binarizer.inverse_transform(male_trans)
print('is_male:', male_trans)
print('ori_class:', gender_class_inve)
# 多值分类也可以
province_label_list = np.array(['河南省', '陕西省', '江苏省', '河北省'])
province_bina_trans = label_binarizer.fit_transform(province_label_list)
# [0 0 1 0] 代表河南省,正常生产例如电商或者物流，会建立映射纬度表，三元结构，province_name,province_code,province_class_lable(34，加台湾为35)
print(province_bina_trans)
