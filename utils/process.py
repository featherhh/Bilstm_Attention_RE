#-*- coding: utf-8 -*- 
# @Time : 2023/2/22 15:05
# @Author : ligang
# coding:utf-8
from config import *
from itertools import chain
import sys

from collections import Counter

conf = Config()


relation2id = {}
with open(conf.rel_data_path, 'r', encoding='utf-8')as fr:
    for line in fr.readlines():
        word, id = line.rstrip().split(' ')
        if word not in relation2id:
            relation2id[word] = id


def sent_padding(words, word2id):
    """把 words 转为 id 形式，并自动补全为 max_len 长度。"""
    ids = []
    # print(f'len(words)的长度--》{len(words)}')
    for word in words:
        if word in word2id:
            ids.append(word2id[word])
        else:
            ids.append(word2id['UNKNOW'])
    # print(f'ids-->{ids}')
    # print(f'ids的长度-->{len(ids)}')
    if len(ids) >= conf.max_len:
        return ids[:conf.max_len]
    ids.extend([word2id['BLANK']]*(conf.max_len-len(ids)))
    return ids


def pos(num):
    if num < -70:
        return 0
    if num >= -70 and num <= 70:
        return num+70
    if num > 70:
        return 142


def position_padding(pos_ids):
    pos_ids = [pos(id) for id in pos_ids]
    if len(pos_ids) >= conf.max_len:
        return pos_ids[:conf.max_len]
    pos_ids.extend([142]*(conf.max_len - len(pos_ids)))
    return pos_ids


def get_txt_data(data_path):
    datas = []
    labels = []
    positionE1 = []
    positionE2 = []
    entities = []
    count_dict = {key: 0 for key, value in relation2id.items()}
    with open(data_path, 'r', encoding='utf-8')as tfr:
        for line in tfr.readlines():
            line = line.rstrip().split(' ', maxsplit=3)
            if line[2] not in count_dict:
                continue
            if count_dict[line[2]] > 2000:
                continue
            else:
                entities.append([line[0], line[1]])
                sentence = []
                index1 = line[3].index(line[0])
                position1 = []
                index2 = line[3].index(line[1])
                position2 = []
                assert len(line) == 4
                for i, word in enumerate(line[3]):
                    sentence.append(word)
                    position1.append(i-index1)
                    position2.append(i-index2)

                datas.append(sentence)
                labels.append(relation2id[line[2]])
                positionE1.append(position1)
                positionE2.append(position2)
                count_dict[line[2]] += 1

    return datas, labels, positionE1, positionE2, entities


def get_word_id(data_path):
    datas, labels, positionE1, positionE2, entities = get_txt_data(data_path)
    data_list = list(set(chain(*datas)))
    word2id = {word: id for id, word in enumerate(data_list)}
    id2word = {id: word for id, word in enumerate(data_list)}
    word2id["BLANK"] = len(word2id)
    word2id["UNKNOW"] = len(word2id)
    id2word[len(id2word)] = "BLANK"
    id2word[len(id2word)] = "UNKNOW"
    return word2id, id2word


if __name__ == '__main__':
    word2id, id2word = get_word_id(data_path=conf.train_data_path)
    print(word2id)
    print(id2word)