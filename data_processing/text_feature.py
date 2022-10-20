from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import torch.nn.functional as F
import jieba
import torch
import numpy as np

dicts = [
    {"name": "tshirt", "price": 90},
    {"name": "coat", "price": 500},
    {"name": "hat", "price": 15}
]
vec = DictVectorizer()
data = vec.fit_transform(dicts).toarray()
print(data)
print(vec.get_feature_names())
print(vec.get_feature_names_out(input_features="name"))

# 001 bag_of_words,词袋模型，大量用于检索，和关联推荐的相似度和距离计算
words = ['I am a developer',
         'I use python and java,I want to study rust',
         'everything is table,everything is graph,everything is network']
# 设置常用停用词,其实就是一些停用词和发语词，将无实际含义的词剔除或者忽略
vectorizers = CountVectorizer(stop_words='english')
word_dense = vectorizers.fit_transform(words).todense()
# 转换稀疏矩阵
word_dense_voc = vectorizers.vocabulary_
print(word_dense)
print(word_dense_voc)
# 002 jieba分词，结巴分词可以担负一些比较常见的场景，专业的分词需要引入特定词库或者类似于阿里云es的领域词库服务（两种模式，一种是专家领域模式，行业词库，另外一种是开发平台自己维护词库做分词累和沉积）
words_chinese = ['php是世界上最好的语言', 'rust有生命周期管理和安全性更高', 'c高级语言性能都不给力，安全也要靠系统']
# i for i in words_chinese 列表推导式，用于元素迭代；
word_cut = ['/'.join(jieba.cut(i)) for i in words_chinese]
print(word_cut)
vectorizer_chin = CountVectorizer()
cnt = vectorizer_chin.fit_transform(word_cut).todense()
print(cnt)
print(vectorizer_chin.vocabulary_)

# 003 文本距离 欧式距离，其实就是坐标对应的几何距离
vectorizers_text = CountVectorizer()
for x, y in [[0, 1], [0, 2], [1, 2]]:
    dist = euclidean_distances(cnt[x], cnt[y])
    # dist = euclidean_distances(str(x), str(y))
    print('文本{}和文本{}之间的距离为{}'.format(x, y, dist))
# 004 相似度计算
vec1_arr = [1, 2, 3, 4, 5]
vec2_arr = [6, 7, 8, 9, 10]
vec1 = torch.FloatTensor(vec1_arr)
vec2 = torch.FloatTensor(vec2_arr)
"""
这里提供余弦相似度的计算公式，cos_sim_np numpy的相似度计算公式，cos_sim_skl sklearnd的计算方式，cos_sim_torch pytorch的计算方式，注意，torch方式计算返回的是一个张量tensor
余弦相似度的计算方式参看numpy的计算:向量间内积/向量的模乘积
"""
cos_sim_np = np.array(vec1_arr).dot(np.array(vec2_arr)) / (np.linalg.norm(np.array(vec1_arr)) * np.linalg.norm(
    np.array(vec2_arr)))
cos_sim_skl = cosine_similarity(np.array(np.array(vec1_arr)).reshape(1, -1), np.array(vec2_arr).reshape(1, -1))
cos_sim_torch = F.cosine_similarity(vec1, vec2, dim=0)
print(cos_sim_np, cos_sim_skl, cos_sim_torch)
"""
0.9649505047327671 
[[0.9649505]] 
tensor(0.9650)
"""
# 权重向量,问题的处理有很多因子或者考虑因素，比如古城西安要修地铁，两个站点的距离，路线需要动迁的补偿款，是否涉及到古遗迹汉墓之类，土层的硬度种种，不同的考虑因素在最终决策需要不同的权重向量或者说因子系数
"""
举例以tf-idf为例，nlp和文本数据挖掘里常用
TF-IDF（term frequency–inverse document frequency，词频-逆向文件频率）是一种用于信息检索（information retrieval）与文本挖掘（text mining）的常用加权技术。
TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。
TF-IDF的主要思想是：如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
引用：https://blog.csdn.net/weixin_43526268/article/details/123262308
"""
texts = ['鲁迅先生写下：我家门前有两棵树，一棵是枣树，另外一棵也是枣树', '枣树是多年生植物，一般需要好多年，才能结枣子',
         '鲁迅写下的文章，关我周树人什么事，我知道中文语法，还不能皮一下了']
texts_cut = ['/'.join(jieba.cut(i)) for i in texts]
print(texts_cut)
# texts_new = []
# for text in texts_cut:
#     texts_new.append(text)
text_exc = CountVectorizer(stop_words=['的', '了']).fit_transform(texts_cut)

print(text_exc)
tfidf = TfidfTransformer().fit_transform(text_exc)
print(tfidf)
