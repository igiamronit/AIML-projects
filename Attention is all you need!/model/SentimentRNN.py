import torch
import torch.nn as nn
import torch.nn.functional as F

class SentimentRNN(nn.Module):
    def __init__(self, embedding_matrix, rnn_type, hidden_size=128, num_layers = 1, bidirectional = False, attention = None, num_classes = 2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention

        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_matrix.shape[1], hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_matrix.shape[1], hidden_size, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout)
        else:
            raise ValueError("rnn_type must be 'RNN' or 'LSTM'")

        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        if self.rnn_type == 'LSTM':
            outputs, (h_n, _) = self.rnn(emb)
        else:
            outputs, h_n = self.rnn(emb)

        if self.attention is not None:
            # Use last hidden state for attention query
            if self.bidirectional:
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                h_last = h_n[-1]
            context, attn_weights = self.attention(outputs, h_last)
            out = self.fc(context)
            return out, attn_weights
        else:
            # Use last hidden state
            if self.bidirectional:
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                h_last = h_n[-1]
            out = self.fc(h_last)
            return out, None