import torch
import torch.nn as nn
from transformers import AutoModel

class MultimodalResumeClassifier(nn.Module):
    def __init__(
            self,
            bert_model_name = 'bert-base-uncased',
            tabular_input_dim = 5,
            bert_hidden_dim = 768,
            tabular_hidden_dim = 128,
            fusion_hidden_dim = 256,
            dropout_rate = 0.3,
            num_classes = 1
    ):
        super(MultimodalResumeClassifier, self).__init__()

        self.bert = AutoModel.from_pretrained(bert_model_name)

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_input_dim, tabular_hidden_dim),
            nn.BatchNorm1d(tabular_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(tabular_hidden_dim, tabular_hidden_dim // 2),
            nn.BatchNorm1d(tabular_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        fusion_input_dim = bert_hidden_dim + (tabular_hidden_dim // 2)
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.BatchNorm1d(fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.BatchNorm1d(fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)  # Output logits (raw scores)
        )

    def forward(self, input_ids, attention_masks, tabular_features):
        bert_outputs = self.bert(input_ids, attention_masks)
        text_features = bert_outputs.pooler_output
        tabular_encoded = self.tabular_encoder(tabular_features)
        fused_features = torch.cat([text_features, tabular_encoded], dim =1)
        fused_output = self.fusion_layer(fused_features)
        logits = self.classifier(fused_output)

        return logits.squeeze(-1)