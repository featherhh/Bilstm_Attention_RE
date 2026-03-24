# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1)


class BiLSTM_ATT(nn.Module):
    def __init__(self, conf, vocab_size, pos_size, tag_size):
        super(BiLSTM_ATT, self).__init__()
        self.batch = conf.batch_size
        self.device = conf.device
        self.vocab_size = vocab_size
        self.embedding_dim = conf.embedding_dim

        self.hidden_dim = conf.hidden_dim

        self.pos_size = pos_size
        self.pos_dim = conf.pos_dim

        self.tag_size = tag_size

        self.word_embeds = nn.Embedding(self.vocab_size,
                                        self.embedding_dim)

        self.pos1_embeds = nn.Embedding(self.pos_size,
                                        self.pos_dim)
        self.pos2_embeds = nn.Embedding(self.pos_size,
                                        self.pos_dim)

        self.lstm = nn.LSTM(input_size=self.embedding_dim + self.pos_dim * 2,
                            hidden_size=self.hidden_dim // 2,
                            num_layers=1,
                            bidirectional=True)

        self.linear = nn.Linear(self.hidden_dim,
                                self.tag_size)

        self.dropout_emb = nn.Dropout(p=0.2)
        self.dropout_lstm = nn.Dropout(p=0.2)
        self.dropout_att = nn.Dropout(p=0.2)

        self.att_weight = nn.Parameter(torch.randn(self.batch, 1, self.hidden_dim).to(self.device))

    def init_hidden_lstm(self):
        return (torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device),
                torch.randn(2, self.batch, self.hidden_dim // 2).to(self.device))

    def attention(self, H):
        # H-->形状【batch_size, hidden_dim, sequence_length】--》[4, 128, 8]
        M = F.tanh(H)
        # a-->形状【batch_size, 1, sequence_length】
        a = F.softmax(torch.bmm(self.att_weight, M), dim=-1)
        # a-->形状【batch_size, sequence_length, 1】-->[4, 8, 1]
        a = torch.transpose(a, 1, 2)

        return torch.bmm(H, a)

    def forward(self, sentence, pos1, pos2):

        init_hidden = self.init_hidden_lstm()

        embeds = torch.cat((self.word_embeds(sentence), self.pos1_embeds(pos1), self.pos2_embeds(pos2)), 2)

        embeds = self.dropout_emb(embeds)
        # embeds-->形状【sequence_length, batch_size, embed_dim】
        embeds = torch.transpose(embeds, 0, 1)

        lstm_out, lstm_hidden = self.lstm(embeds, init_hidden)
        # lstm_out-->形状【batch_size, hidden_dim, sequence_length】
        lstm_out = lstm_out.permute(1, 2, 0)

        lstm_out = self.dropout_lstm(lstm_out)
        att_out = F.tanh(self.attention(lstm_out))
        att_out = self.dropout_att(att_out).squeeze()

        result = self.linear(att_out)

        return result
