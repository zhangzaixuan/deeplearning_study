from sklearn.feature_extraction import DictVectorizer
import jieba
from sklearn.metrics.pairwise import euclidean_distances

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
from sklearn.feature_extraction.text import CountVectorizer

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
# jieba分词，结巴分词可以担负一些比较常见的场景，专业的分词需要引入特定词库或者类似于阿里云es的领域词库服务（两种模式，一种是专家领域模式，行业词库，另外一种是开发平台自己维护词库做分词累和沉积）
words_chinese = ['php是世界上最好的语言', 'rust有生命周期管理和安全性更高', 'c高级语言性能都不给力，安全也要靠系统']
# i for i in words_chinese 列表推导式，用于元素迭代；
word_cut = ['/'.join(jieba.cut(i)) for i in words_chinese]
print(word_cut)
vectorizer_chin = CountVectorizer()
cnt = vectorizer_chin.fit_transform(word_cut).todense()
print(cnt)
print(vectorizer_chin.vocabulary_)

# 文本距离 欧式距离，其实就是坐标对应的几何距离
vectorizers_text = CountVectorizer()
for x, y in [[0, 1], [0, 2], [1, 2]]:
    dist = euclidean_distances(cnt[x], cnt[y])
    # dist = euclidean_distances(str(x), str(y))
    print('文本{}和文本{}之间的距离为{}'.format(x, y, dist))
