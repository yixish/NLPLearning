import collections
import os
import random
import time
from tqdm import tqdm
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd

index_dict = {'咨询': 0,'表扬': 1,  '建议': 2, '投诉': 3, '其他': 4}


def read_data(path):
    train_df = pd.read_excel(path)

    keys = list(train_df['问题'].values)
    vals = list(train_df['分类'].values)
    vals = list(map(lambda x: index_dict[x], vals))
    data = []
    max_l = -1
    for i in range(len(keys)):
        if len(keys[i]) > max_l:
            max_l = len(keys[i])
        data.append([keys[i], vals[i]])
    data_len = len(data)
    print("max_len :"+str(max_l))
    return data[:int(data_len * 0.8)], data[int(data_len * 0.2):]


def get_tokenized(data):
    '''
    @params:
        data: 数据的列表，列表中的每个元素为 [文本字符串，0/1标签] 二元组
    @return: 切分词后的文本的列表，列表中的每个元素为切分后的词序列
    '''

    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]

    return [tokenizer(review) for review, _ in data]


def get_vocab(data):
    '''
    @params:
        data: 同上
    @return: 数据集上的词典，Vocab 的实例（freqs, stoi, itos）
    '''
    tokenized_data = get_tokenized(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return Vocab.Vocab(counter, min_freq=1)


max_l = 100  # 将每条评论通过截断或者补0，使得长度变成500


def preprocess(data, vocab):
    '''
    @params:
        data: 同上，原始的读入数据
        vocab: 训练集上生成的词典
    @return:
        features: 单词下标序列，形状为 (n, max_l) 的整数张量
        labels: 情感标签，形状为 (n,) 的0/1整数张量
    '''

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized(data)
    features = torch.tensor([pad([vocab.stoi[word] for word in words]) for words in tokenized_data])
    labels = torch.tensor([score for _, score in data])
    return features, labels


class TextRNN(nn.Module):

    def __init__(self, vocab, embedding_dim, output_dim, hidden_size, num_layers, bidirectional, dropout):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocab), embedding_dim)
        # self.embedding = nn.Embedding.from_pretrained(
        #     pretrained_embeddings, freeze=False)
        self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, bidirectional=bidirectional, dropout=dropout)
        self.W2 = nn.Linear(2 * hidden_size + embedding_dim, hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        text = x.T
        # text: [seq_len, batch size]
        embedded = self.dropout(self.embedding(text))
        # embedded: [seq_len, batch size, emb dim]

        outputs, _ = self.rnn(embedded)
        # outputs: [seq_len， batch_size, hidden_size * bidirectional]

        outputs = outputs.permute(1, 0, 2)
        # outputs: [batch_size, seq_len, hidden_size * bidirectional]

        embedded = embedded.permute(1, 0, 2)
        # embeded: [batch_size, seq_len, embeding_dim]

        x = torch.cat((outputs, embedded), 2)
        # x: [batch_size, seq_len, embdding_dim + hidden_size * bidirectional]

        y2 = torch.tanh(self.W2(x)).permute(0, 2, 1)
        # y2: [batch_size, hidden_size * bidirectional, seq_len]

        y3 = F.max_pool1d(y2, y2.size()[2]).squeeze(2)
        # y3: [batch_size, hidden_size * bidirectional]

        return self.fc(y3)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval()
                yy = net(X.to(device)).argmax(dim=1)
                acc_sum += ((yy) == y.to(device)).float().sum().cpu().item()
                net.train()
            else:
                if ('is_training' in net.__code__.co_varnames):
                    yy = net(X, is_training=False).argmax(dim=1)
                    acc_sum += (yy == y).float().sum().item()
                else:
                    yy = net(X).argmax(dim=1) == y
                    acc_sum += yy.float().sum().item()
            n += y.shape[0]
            # print(yy)
    return acc_sum / n


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


if __name__ == '__main__':
    train_data, test_data = read_data("data.xls")
    vocab = get_vocab(train_data)
    print('# words in vocab:', len(vocab))
    train_set = Data.TensorDataset(*preprocess(train_data, vocab))
    test_set = Data.TensorDataset(*preprocess(test_data, vocab))

    batch_size = 16
    train_iter = Data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = Data.DataLoader(test_set, batch_size)

    for X, y in train_iter:
        print('X', X.shape, 'y', y.shape)
        break
    print('#batches:', len(train_iter))
    embed_size, num_hiddens, num_layers = 50, 50, 2
    out_dim = 5
    model = TextRNN(vocab, embed_size, out_dim, num_hiddens, num_layers, True, 0.5)
    # cache_dir = "/content/gdrive/My Drive/dataset/GloVe6B"
    # glove_vocab = Vocab.GloVe(name='6B', dim=100, cache=cache_dir)
    lr, num_epochs = 0.01, 50
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    loss = nn.CrossEntropyLoss()
    train(train_iter, test_iter, model, loss, optimizer, device, num_epochs)
