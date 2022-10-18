import pandas as pd
import numpy as np

score = {'active': [98, 85, 30, 66, 88, 99, 80], 'gift': [98, 85, 30, 66, 88, 99, 50],
         'fans': [98, 85, 30, 66, 88, 99, 80]}
df = pd.DataFrame(score, index=['kelaoban', 'wenlaoban', 'ruchao', 'yaru', 'houfeng', 'lidan', 'zaixuan'])
# print(df)
# 主要介绍pandas的api,牵涉到多数据源的时候，建议用pands来做程序的数据中间层，例如mysql和spark,pandas内部的api效率相对高一点，自己操作数据容易忘记列表迭代式这些相对优化的写法
# 001 drop 删除行或者列， index:列，columns:行
df1 = df.drop(index=['zaixuan'])
print(df1)
df2 = df.drop(columns=['gift'])
print(df2)
print(df)
# 002 重命名列名，两个rename的操作基本一致，都是替换列名；inplace 为false是不替换原有数据dataframe,而是生成替换列名后的数据副本，为true时替换原有数据
df_rename_reply = df.rename(columns={'active': 'login', 'gift': 'coinordiamond'}, inplace=False)
df_rename = df.rename(columns={'active': 'login', 'gift': 'coinordiamond'}, inplace=True)
print(df_rename_reply)
print(df)
# 003 去掉重复值
"""
subset column label or sequence of labels
subset: Hashable | Sequence[Hashable] | None = None,
keep: Literal["first", "last", False] = "first",
subset
"""

df_example = pd.DataFrame({
    'brand': ['Yum Yum', 'Yum Yum', 'Indomie', 'Indomie', 'Indomie'],
    'style': ['cup', 'cup', 'cup', 'pack', 'pack'],
    'rating': [4, 4, 3.5, 15, 5]
})
df_dupl = df_example.duplicated()
df_dupl_result = df_example.drop_duplicates()
print(df_example, "\n", df_dupl, df_dupl_result)
"""
第二列表示该列值是不是重复值，判断依据要依赖keep,默认为first,就是两数据（这里是行）重复，保留前面的，后边的被认定为重复值
0    False
1     True
2    False
3    False
4    False
dtype: bool
"""
# 004 转换数据格式 默认需要生成新的数据集
df_type_convert = df['login'].astype('str')
df['fans'].astype(np.float64)
print(df.head(3))
# FutureWarning: iteritems is deprecated and will be removed in a future version. Use .items instead
# print(df_type_convert.head(1), type(df_type_convert.head(1)), type(df_type_convert.head(1)['login']) )
print(df_type_convert.head(1), type(df_type_convert.head(1)), )

for i in df_type_convert.head(1):
    print(i, type(i))
"""
kelaoban    98
Name: login, dtype: object <class 'pandas.core.series.Series'>
98 <class 'str'>
"""
# 填充缺省值
df_emp_ori = pd.DataFrame([[np.nan, 3, 0], [2, np.nan, 0], [1, 0, np.nan]], columns=list('abc'))
print(df_emp_ori)
print(df_emp_ori.isnull)
print(df_emp_ori.fillna(0))
# df_emp_ori.ffill(),等效下面的填充
print(df_emp_ori.fillna(method='ffill'))
# 均值填充
print(df_emp_ori.fillna(df_emp_ori.mean()))
values = {'a': 1, 'b': 2, 'c': 3}
print(df_emp_ori.fillna(value=values))
