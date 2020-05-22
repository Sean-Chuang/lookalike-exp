#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import random
from collections import Counter
 
 
class SGNS(nn.Module):
 
    def __init__(self, vocab_size, projection_dim):
 
        super(SGNS, self).__init__()
        # Not shared same embedding
        self.embedding_v = nn.Embedding(vocab_size, projection_dim)     # center embedding
        self.embedding_u = nn.Embedding(vocab_size, projection_dim)     # out embedding
        self.log_sigmoid = nn.LogSigmoid()
 
        init_range = (2.0 / (vocab_size + projection_dim)) ** 0.5       # Xavier init
        self.embedding_v.weight.data.uniform_(-init_range, init_range)      # init
        self.embedding_u.weight.data.uniform_(-init_range, init_range)  # init
 
    def forward(self, center_words, target_words, negative_words):
 
        center_embeds = self.embedding_v(center_words)      # B * 1 * D
 
        target_embeds = self.embedding_u(target_words)      # B * 1 * D
        neg_embeds = -self.embedding_u(negative_words)      # B * K * D
 
        positive_score = target_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2)        # B * 1
        negative_score = torch.sum(neg_embeds.bmm(center_embeds.transpose(1, 2)).squeeze(2), 1).view(center_words.size(0), -1)      # B * K ---> B * 1
 
        los = self.log_sigmoid(positive_score) + self.log_sigmoid(negative_score)
 
        return -torch.mean(los)
 
    # 获取单词的embedding
    def prediction(self, inputs):
 
        embeds = self.embedding_v(inputs)
 
        return embeds
 
 
def get_batch_sample(bat_size, tra_data):       # ~
 
    random.shuffle(tra_data)
 
    s_index = 0
    e_index = bat_size
 
    while e_index < len(tra_data):
        bat = tra_data[s_index: e_index]
        temp = e_index
        e_index = e_index + bat_size
        s_index = temp
        yield bat
 
    if e_index >= len(tra_data):
        bat = train_data[s_index:]
        yield bat
 
 
def get_positive_sample(samp_lists):        # ~
    """
    :param :list: 二维列表
    :return:
    """
 
    positive_samples = []
 
    for sublist in samp_lists:
 
        sublist_length = len(sublist)
 
        for ite in sublist:
 
            ite_index = sublist.index(ite)
 
            for j in range(sublist_length):
 
                if ite_index != j:
                    positive_samples.append([ite, sublist[j]])
 
    target_words = []
    context_words = []
 
    for word_pair in positive_samples:
        target_words.append(word_pair[0])
        context_words.append(word_pair[1])
 
    return target_words, context_words      # 一维列表
 
 
def get_negative_sample(centers, targets, un_table, k):
 
    batch_size = len(targets)       # 批次大小
 
    negative_samples = []
 
    for i in range(batch_size):
 
        neg_sample = []
        center_index = centers[i][0]        # !!!
        target_index = targets[i][0]        # !!!
 
        while len(neg_sample) < k:
 
            neg = random.choice(un_table)
            if neg == target_index or neg == center_index:
                continue
            neg_sample.append(neg)
        negative_samples.append(neg_sample)
 
    # 返回一个二维列表
    return negative_samples
 
 
if __name__ == '__main__':
 
    movie_lists = []
 
    with open(r'E:\Experiment\Algorithms\Item2vec-pytorch\doulist_0804_09.movie_id', 'r', encoding='utf8') as f:
        contents = f.readlines()
        for content in contents:
            content = content.strip().split(' ')
            if content[0] == '':
                continue
            movie_list = [int(m) for m in content]
            if len(movie_list) > 1:
                movie_lists.append(movie_list)      # 二维列表
 
    fla = lambda k: [i for sublist in k for i in sublist]
    item_counter = Counter(fla(movie_lists))
    item = [w for w, c in item_counter.items()]  # item列表
 
    item2index = {}
    for vo in item:
        if item2index.get(vo) is None:
            item2index[vo] = len(item2index)
 
    index2word = {v: k for k, v in item2index.items()}
 
    new_movie_lists = []
    for m in movie_lists:
        m = [item2index[n] for n in m]
        new_movie_lists.append(m)
 
    cent_words, cont_words = get_positive_sample(new_movie_lists)  # 一维列表
 
    uni_table = []
    f = sum([item_counter[it] ** 0.75 for it in item])
    z = 0.0001
    for it in item:
        uni_table.extend([it] * int(((item_counter[it] ** 0.75) / f) / z))
 
    train_data = [[cent_words[i], cont_words[i]] for i in range(len(cent_words))]       # 二维列表
 
    item2vec = SGNS(len(item), 10)      # ~
    print(item2vec)
    optimizer = optim.Adam(item2vec.parameters(), lr=0.001)
 
    for epoch in range(10):
 
        for i, batch in enumerate(get_batch_sample(2048, train_data)):
 
            target = [[p[0]] for p in batch]
            context = [[q[1]] for q in batch]
            negative = get_negative_sample(centers=target, targets=context, un_table=uni_table, k=10)
 
            target = Variable(torch.LongTensor(target))
            # print(target)
            context = Variable(torch.LongTensor(context))
            # print(context)
            negative = Variable(torch.LongTensor(negative))
            # print(negative)
            item2vec.zero_grad()
 
            loss = item2vec(target, context, negative)
 
            loss.backward()
            optimizer.step()
 
            print('Epoch : %d, Batch : %d, loss : %.04f' % (epoch + 1, i + 1, loss))