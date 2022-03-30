#onehot
# use keras to generate onehot
from numpy import array,argmax
from keras import to_categorical
# 读取数据
def load_data(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    data = [line.strip().split('\t')[1] for line in data][1:]
    return data

# 获取所有的字符
all_char=list(set(''.join(load_data('../data/toutiaonews38w/test.tsv'))))[:100]
values=array(all_char)
print(values)
# # 数值编码
# label_encoder=LabelEncoder()
# integer_encoded=label_encoder.fit_transform(values)

# # one-hot编码
# encoded=to_categorical(integer_encoded)
# print(encoded)

