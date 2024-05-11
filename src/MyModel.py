import torch
from torch import nn
import torch.optim
from TorchCRF import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags)
        
    def forward(self, input_ids, target=None):
        embeds = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        emissions = self.linear(lstm_out)
        if target is not None:
            return -self.crf(emissions, target, reduction='mean')
        else:
            return self.crf.decode(emissions)