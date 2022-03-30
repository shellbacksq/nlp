import numpy as np

# 读取数据
def load_data(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    data = [line.strip().split('\t')[1] for line in data][1:]
    return data

# 获取所有的字符
all_char=list(set(''.join(load_data('../data/toutiaonews38w/test.tsv'))))

# 建立映射
char2id={char:id for id,char in enumerate(all_char)}
id2char={id:char for id,char in enumerate(all_char)}

def one_hot(data):
    data_id=[char2id[char] for char in data]
    data_matrix=np.zeros((len(data),len(all_char)))
    for i,id in enumerate(data_id):
        data_matrix[i,id]=1
    return data_matrix


if __name__ == '__main__':
    for data in load_data('../data/toutiaonews38w/test.tsv')[:10]:
        print(one_hot(data))