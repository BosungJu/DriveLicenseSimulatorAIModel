"""LSTM network for variable-length section sequences."""

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class SectionLSTM(nn.Module):
    def __init__(self, feature_count, label_count, hidden_size=64, num_layers=1):
        super().__init__()
        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        self.lstm = nn.LSTM(feature_count, hidden_size, num_layers=num_layers, batch_first=True)
        self.classifier = nn.Linear(hidden_size, label_count)

    def forward(self, features, lengths):
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.classifier(hidden[-1])
