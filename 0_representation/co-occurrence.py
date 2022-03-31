import numpy as np
import jieba
from collections import defaultdict

# 读取数据,分词
def load_data(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    data = [jieba.lcut(line.strip().split('\t')[1]) for line in data][1:]
    return data[:2]
all_line=load_data('../data/toutiaonews38w/test.tsv')

all_words=list(set([w for l in all_line for w in l]))
word2id={w:i for i,w in enumerate(all_words)}
id2word={i:w for i,w in enumerate(all_words)}

print(all_words)

def build_co_occurrence_matrix(data):
    """
    计算词频矩阵
    """
    co_occurrence_matrix=np.zeros((len(all_words),len(all_words)))
    for line in data:
        for i in range(len(line)):
            for j in range(i+1,len(line)):
                co_occurrence_matrix[word2id[line[i]]][word2id[line[j]]]+=1
    return co_occurrence_matrix

if __name__=='__main__':
    co_occurrence_matrix=build_co_occurrence_matrix(all_line)
    # print(co_occurrence_matrix)
    U,s,v=np.linalg.svd(co_occurrence_matrix,full_matrices=False)
    print(U)