import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class AttentionPool(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden_dim]
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(attn_weights * lstm_out, dim=1)       # [batch, hidden_dim]
        return context


#Without Layer normalization
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=18, hidden_dim=512, num_layers=2, num_classes=2002, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=bidirectional,
        )
        self.attn_pool = AttentionPool(hidden_dim * (2 if bidirectional else 1))
        self.classifier = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x, lengths):  # x: [batch, seq_len, input_dim], lengths: [batch]
        # Pack padded batch of sequences
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pass through LSTM
        packed_output, _ = self.lstm(packed)

        # Unpack sequence
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        # lstm_out: [batch, seq_len, hidden_dim * num_directions]

        # Attention pooling (works on full output, but masking is skipped here for simplicity)
        pooled = self.attn_pool(lstm_out)  # [batch, hidden_dim * 2]

        # Final classification
        logits = self.classifier(pooled)   # [batch, num_classes]
        return logits



# With Layer normalization

# class LSTMClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, num_classes, bidirectional=True, dropout=0.3):
#         super(LSTMClassifier, self).__init__()
#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, 
#                             batch_first=True, dropout=dropout, bidirectional=bidirectional)
#         self.layer_norm = nn.LayerNorm(hidden_dim * 2 if bidirectional else hidden_dim)
#         self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)

#     def forward(self, x, lengths):
#         packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
#         packed_out, _ = self.lstm(packed)
#         output, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
#         # Get the last relevant output (like before)
#         idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(output.size(0), 1, output.size(2))
#         last_output = output.gather(1, idx).squeeze(1)

#         # Apply LayerNorm
#         last_output = self.layer_norm(last_output)
        
#         return self.fc(last_output)

