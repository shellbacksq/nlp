import torch
import jieba
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tensorboardX import SummaryWriter
from tqdm import tqdm

# 文本加载器


class DataSet(Dataset):
    """
    1. 读取文本，做预处理
    2. 返回语料库大小
    3. 返回context和target
    """

    def __init__(self, file_path, ngram=5):
        self.data = []
        with open(file_path, 'r') as f:
            self.data = f.readlines()
        self.data = [jieba.lcut(line.strip().split('\t')[1])
                     for line in self.data][1:]
        self.grams = []
        for sent in self.data:
            self.sent_gram = [([sent[i+j] for j in range(ngram-1)], sent[i + ngram - 1])
                              for i in range(len(sent) - ngram + 1)]
            self.grams.extend(self.sent_gram)
        self.vocab = Counter([w for l in self.data for w in l])
        self.word2id = {word_tuple[0]: idx for idx,
                        word_tuple in enumerate(self.vocab.most_common())}
        self.vocab_size = len(self.vocab)
        self.ngram = ngram

    def __getitem__(self, idx):
        context = torch.tensor([self.word2id[w]
                               for w in self.grams[idx][0]])
        target = torch.tensor([self.word2id[self.grams[idx][1]]])
        return context, target

    def __len__(self):
        return len(self.grams)

# 定义模型


class NgramLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_gram):
        super(NgramLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        self.linear1 = nn.Linear((n_gram-1)*emb_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs, batch_size):
        emb = self.embeddings(inputs).view(batch_size, -1)  # 将inputs转换为emb并拼接起来
        out = F.relu(self.linear1(emb))  # 非线性映射
        out = self.linear2(out)  # 线性映射到词表大小
        log_probs = F.log_softmax(out, dim=1)  # 计算log_softmax
        return log_probs


# 开始训练
ngram = 5
batch_size = 400
embedding_dim = 50
num_epoch=10

data_path="../data/toutiaonews38w/test.tsv"
data=DataSet(data_path,ngram)
model=NgramLanguageModel(data.vocab_size,embedding_dim,ngram)
optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
loss_function=nn.NLLLoss()
losses=[]
cuda_available=torch.cuda.is_available()
data_loader=DataLoader(data,batch_size=batch_size,shuffle=True)

writer=SummaryWriter("../logs/NNLM")

for epoch in range(num_epoch):
    total_loss=0
    for context,target in tqdm(data_loader):
        if context.size()[0]!=batch_size:
            continue
        if cuda_available:
            context=context.cuda()
            target=target.squeeze(1).cuda()
            model=model.cuda()
        model.zero_grad()
        log_prob=model(context,batch_size)
        loss=loss_function(log_prob,target)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    
    losses.append(total_loss)
    writer.add_scalar("loss",total_loss,epoch)

    print("epoch:",epoch,"loss:",total_loss)

writer.close()
embed_matrix = model.embeddings.weight.detach().cpu().numpy()
print(embed_matrix[:10])

        



