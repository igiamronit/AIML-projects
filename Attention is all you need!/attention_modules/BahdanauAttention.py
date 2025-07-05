import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size

        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)

        self.V = nn.Linear(hidden_size,1)

    def forward(self, encoder_outputs, decoder_hidden):
        #encoder output should be like (batchsize x seq_len x hidden_size)
        # decoder hidden should be like (batch_size x hidden_size)

        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1,seq_len,1)

        score = self.V(torch.tanh(self.W1(decoder_hidden)+self.W2(encoder_outputs)))

        attention_weights = F.softmax(score, dim=1).squeeze(2)

        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)

        return context_vector, attention_weights