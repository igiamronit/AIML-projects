import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongDotAttention(nn.Module):
    def __init__(self):
        super(LuongDotAttention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(2)

        score = torch.bmm(encoder_outputs, decoder_hidden)
        attention_weights = F.softmax(score.squeeze(2), dim =1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)

        return context_vector, attention_weights