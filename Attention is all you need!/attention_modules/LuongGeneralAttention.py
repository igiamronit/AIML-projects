import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongGeneralAttention(nn.Module):
    def __init__(self, hidden_size):
        super(LuongGeneralAttention, self).__init__()
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        transformed = self.W(encoder_outputs)
        decoder_hidden = decoder_hidden.unsqueeze(2)
        score = torch.bmm(transformed, decoder_hidden)
        attention_weights = torch.softmax(score.squeeze(2), dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context_vector, attention_weights