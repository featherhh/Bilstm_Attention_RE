# coding:utf-8
import torch


class Config(object):
    def __init__(self):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'mps'
        self.train_data_path = '/Users/ligang/PycharmProjects/NLP/Relation_Extraction/Bilstm_Attention_RE/data/train.txt'
        self.test_data_path = '/Users/ligang/PycharmProjects/NLP/Relation_Extraction/Bilstm_Attention_RE/data/test.txt'
        self.rel_data_path = '/Users/ligang/PycharmProjects/NLP/Relation_Extraction/Bilstm_Attention_RE/data/relation2id.txt'
        self.embedding_dim = 128
        self.pos_dim = 32
        self.hidden_dim = 200
        self.epochs = 50
        self.batch_size = 64
        self.max_len = 70
        self.learning_rate = 1e-3


if __name__ == '__main__':
    con = Config()
    print(con.rel_data_path)