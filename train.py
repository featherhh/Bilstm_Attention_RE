# coding:utf-8
from model.bilstm_atten import *
from utils.data_loader import *
from utils.process import *
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm


def train(conf, vocab_size, pos_size, tag_size):
    # 加载数据集
    train_iter, test_iter = get_loader_data()
    print('训练数据集长度', len(train_iter))
    # 实例化Bilstm+attention模型
    ba_model = BiLSTM_ATT(conf, vocab_size, pos_size, tag_size).to(conf.device)
    # print(ba_model)
    # 实例化优化器
    optimizer = optim.Adam(ba_model.parameters(), lr=conf.learning_rate)

    # 实例化损失函数
    criterion = nn.CrossEntropyLoss()

    # 实现模型训练

    # 定义训练模型参数
    start_time = time.time()
    train_loss = 0  # 已经训练样本的损失
    train_acc = 0  # 已经训练样本的准确率
    total_iter_num = 0  # 训练迭代次数
    total_sample = 0 # 已经训练的样本数


    # 开始模型的训练

    for epoch in range(conf.epochs):
        for sentence, pos1, pos2, label, _, _, _ in tqdm(train_iter):
            # 将数据输入模型
            output = ba_model(sentence, pos1, pos2)

            # 计算损失
            loss = criterion(output, label)

            # 梯度清零
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 梯度更新
            optimizer.step()

            # 计算总损失
            total_iter_num += 1
            train_loss += loss.item()

            # 计算总准确率
            train_acc = train_acc + sum(torch.argmax(output, dim=1) == label).item()

            total_sample = total_sample + label.size()[0]
            # print(f'total_sample--->{total_sample}')

            # 每25次训练，打印日志
            if total_iter_num % 25 == 0:
                tmploss = train_loss / total_iter_num
                tmpacc = train_acc / total_sample
                end_time = time.time()
                print('轮次：%d, 损失:%.6f, 时间:%d, 准确率:%.3f' % (epoch+1, tmploss, end_time-start_time, tmpacc))
        if epoch % 10 == 0:
            torch.save(ba_model.state_dict(), './save_model/new_model_%d.bin' % epoch)


if __name__ == '__main__':
    word2id, id2word = get_word_id(conf.train_data_path)
    vocab_size = len(word2id)
    # print(vocab_size)
    pos_size = 143
    tag_size = len(relation2id)
    print(conf.train_data_path)

    train(conf, vocab_size, pos_size, tag_size)


