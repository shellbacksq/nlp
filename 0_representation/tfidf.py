"""
tf-idf
1. 分词然后获取词表
2. 统计每个词的idf, 即对于一个词，有多少个文档中出现了
3. 在每个文本中统计词频，然后计算tf
4. 对于每个文本计算tf-idf

"""
import numpy as np
import jieba
from collections import defaultdict

# 读取数据,分词
def load_data(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    data = [jieba.lcut(line.strip().split('\t')[1]) for line in data][1:]
    return data[:20]
all_line=load_data('../data/toutiaonews38w/test.tsv')

all_words=list(set([w for l in all_line for w in l]))
word2id={w:i for i,w in enumerate(all_words)}

# 计算idf
idf_dict=defaultdict(int)
for line in all_line:
    for w in line:
        idf_dict[w]+=1

# 计算tf-idf
tfidf_list=[]
for line in all_line:
    tf_dict=defaultdict(int)
    for w in line:
        tf_dict[w]+=1
    tfidf_dict={}
    for w in tf_dict:
        tfidf_dict[w]=tf_dict[w]*np.log(len(all_line)/idf_dict[w])
    tfidf_list.append(tfidf_dict)

# 转化成tf-idf向量
for tfidf in tfidf_list:
    data_array=np.zeros(len(all_words))
    for w in tfidf:
        data_array[word2id[w]]=tfidf[w]
    print(data_array)

if __name__=='__main__':
    print()