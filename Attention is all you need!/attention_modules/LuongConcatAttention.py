import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongConcatAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongConcatAttention, self).__init__()
        self.W = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        concat = torch.cat((encoder_outputs, decoder_hidden), dim=2)
        score = self.v(torch.tanh(self.W(concat)))
        attention_weights = torch.softmax(score.squeeze(2), dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector, attention_weights