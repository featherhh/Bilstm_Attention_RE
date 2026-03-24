# coding:utf-8
import os
from torch.utils.data import DataLoader, Dataset
from utils.process import *
import torch
from rich import print

word2id, id2word = get_word_id(conf.train_data_path)


class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data = get_txt_data(data_path)

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, index):
        sequence = self.data[0][index]
        label = int(self.data[1][index])
        positionE1 = self.data[2][index]
        positionE2 = self.data[3][index]
        entites = self.data[4][index]
        return sequence, label, positionE1, positionE2, entites


def collate_fn(datas):
    # print(len(datas))
    # print(f'datas-->{datas}')
    # print(f'datas取出一个样本-->{datas[0]}')
    sequences = [data[0] for data in datas]
    labels = [data[1] for data in datas]
    positionE1 = [data[2] for data in datas]
    positionE2 = [data[3] for data in datas]
    entities = [data[4] for data in datas]
    # print(f'*'*80)
    sequences_ids = []
    for words in sequences:
        # print(f'words--》{words}')
        ids = sent_padding(words, word2id)
        sequences_ids.append(ids)
    positionE1_ids = []
    positionE2_ids = []
    for pos_ids in positionE1:
        # print(f'pos_ids---》{pos_ids}')
        pos1_ids = position_padding(pos_ids)
        # print(f'处理后的pos_ids---》{pos1_ids}')
        positionE1_ids.append(pos1_ids)
    for pos_ids in positionE2:
        pos2_ids = position_padding(pos_ids)
        positionE2_ids.append(pos2_ids)

    datas_tensor = torch.tensor(sequences_ids, dtype=torch.long, device=conf.device)
    positionE1_tensor = torch.tensor(positionE1_ids, dtype=torch.long, device=conf.device)
    positionE2_tensor = torch.tensor(positionE2_ids, dtype=torch.long, device=conf.device)
    labels_tensor = torch.tensor(labels, dtype=torch.long, device=conf.device)
    return datas_tensor, positionE1_tensor, positionE2_tensor, labels_tensor, sequences, labels, entities


def get_loader_data():
    train_data = MyDataset(conf.train_data_path)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=conf.batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn,
                                  drop_last=True)
    # for value in train_dataloader:
    #     print('你好')
    #     break
    test_data = MyDataset(conf.test_data_path)

    test_dataloader = DataLoader(dataset=test_data,
                                  batch_size=conf.batch_size,
                                  shuffle=False,
                                  collate_fn=collate_fn,
                                  drop_last=True)

    return train_dataloader, test_dataloader



if __name__ == '__main__':
    get_loader_data()