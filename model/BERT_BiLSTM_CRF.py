import torch.nn as nn
import torch
from transformers import BertModel
from torchcrf import CRF
from TCN import TemporalConvNet
from attention import MultiHeadedAttention
from cnn import IDCNN
from torch.autograd import Variable
import numpy as np

class BERT_BiLSTM_CRF(nn.Module):
    def __init__(self, tagset_size, embedding_dim, hidden_dim, rnn_layers, filters_number, dropout, pretrain_model_name,device):
        '''
        the model of BERT_BiLSTM_CRF
        :param bert_config:
        :param tagset_size:
        :param embedding_dim:
        :param hidden_dim:
        :param rnn_layers:
        :param lstm_dropout:
        :param dropout:
        :param use_cuda:
        :return:
        '''
        super(BERT_BiLSTM_CRF, self).__init__()
        self.tagset_size = tagset_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.filters_number = filters_number
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.device = device
        self.word_embeds = BertModel.from_pretrained(pretrain_model_name)
        for param in self.word_embeds.parameters():
            param.requires_grad = True
        self.LSTM = nn.LSTM(768,
                            self.hidden_dim,
                            num_layers=self.rnn_layers,
                            bidirectional=True,
                            batch_first=True)#bilstm/idcnn   bilstm-idcnn

        self._dropout = nn.Dropout(p=self.dropout)
        self.CRF = CRF(num_tags=self.tagset_size, batch_first=True)
        self.Liner = nn.Linear(self.hidden_dim*2, self.tagset_size)
        # self.embedding = nn.Embedding(embedding_dim=embedding_dim,num_embeddings=21128)

        # self.position_embedding = Positional_Encoding(embedding_dim,self.hidden_dim,self.dropout,self.device)

        self.liner = nn.Linear(self.embedding_dim,self.hidden_dim*2)
        # self.TCN = TemporalConvNet(self.hidden_dim*2,[self.filters_number,self.filters_number])
        # self.liner2 = nn.Linear(self.filters_number,self.hidden_dim*2)
        # self.attention = MultiHeadedAttention(self.hidden_dim*2,8)
        # self.linear = nn.Linear(self.embedding_dim,self.tagset_size)
        # self.cnn = IDCNN(self.hidden_dim*2,filters=self.filters_number)

    def _init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return (torch.randn(2*self.rnn_layers, batch_size, self.hidden_dim).to(self.device), \
                torch.randn(2*self.rnn_layers, batch_size, self.hidden_dim).to(self.device))


    def forward(self, sentence, attention_mask=None):
        '''
        :param sentence: sentence (batch_size, max_seq_len) : word-level representation of sentence
        :param attention_mask:
        :return: List of list containing the best tag sequence for each batch.
        '''
        batch_size = sentence.size(0)  #16
        seq_length = sentence.size(1) #128
        # embeds: [batch_size, max_seq_length, embedding_dim]
        embeds = self.word_embeds(sentence, attention_mask=attention_mask).last_hidden_state  # 16 128 768    sentence维度为16  128

        # embeds_word = self.embedding(sentence)
        # embeds_word = self.position_embedding(embeds_word)
        # embeds_word = self.liner(embeds_word)
        # embeds = self.liner(embeds)
        # hidden = self._init_hidden(batch_size)
        # lstm_out: [batch_size, max_seq_length, hidden_dim*2]
        # lstm_out, hidden = self.LSTM(embeds_word)  # 16  128  256

        # cnn_word  = self.TCN(embeds_word.permute(1,2, 0))
        # con_emb = torch.cat((cnn_word,embeds),dim=-1)

        # con_emb = self.liner2(cnn_word.permute(2,0,1))

        # embeds_word = self.attention(con_emb,embeds,embeds)
        hidden = self._init_hidden(batch_size)
        # lstm_out: [batch_size, max_seq_length, hidden_dim*2]
        lstm_out, hidden = self.LSTM(embeds,hidden)  # 16  128  256
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim*2)    # 2048  256
        d_lstm_out = self._dropout(lstm_out)  # 2048  256
        l_out = self.Liner(d_lstm_out)  # 2048  22
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)   # 16  128 22

        return lstm_feats

    def loss(self, feats, tags, mask):
        ''' 做训练时用
        :param feats: the output of BiLSTM and Liner
        :param tags:
        :param mask:
        :return:
        '''
        loss_value = self.CRF(emissions=feats,
                              tags=tags,
                              mask=mask,
                              reduction='mean')
        return -loss_value

    def predict(self, feats, attention_mask):
        # 做验证和测试时用
        out_path = self.CRF.decode(emissions=feats, mask=attention_mask)
        return out_path


class Positional_Encoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int = 32, dropout: int = 0.0, device: str = "cpu"):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([
            [pos / (10000 ** (i / d_model)) if i % 2 == 0 else pos / (10000 ** ((i - 1) / d_model)) for i in
             range(d_model)] for pos in range(max_seq_len)
        ])
        # self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / d_model)) for i in range(d_model)] for pos in range(max_seq_len)])#与上面等价
        # 偶数维度用sin 奇数维度用cos
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x.shape: (batch_size, max_seq_len, d_model)
        # self.pe: (max_seq_len, d_model)
        # 广播机制
        # out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = x + Variable(self.pe, requires_grad=False).to(self.device)  # 与上面等价
        out = self.dropout(out)
        return out