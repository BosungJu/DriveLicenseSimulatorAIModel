"""LSTM network for variable-length section sequences."""

import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# "last" reads the final hidden state; "mean_max" concatenates the mean and max of the outputs over valid time steps.
POOLING_MODES = ("last", "mean_max")


class SectionLSTM(nn.Module):
    def __init__(self, feature_count, label_count, hidden_size=64, num_layers=1, dropout=0.0, pooling="last"):
        super().__init__()
        if type(num_layers) is not int or num_layers <= 0:
            raise ValueError("num_layers must be a positive integer")
        if type(dropout) not in (int, float) or not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if pooling not in POOLING_MODES:
            raise ValueError(f"pooling must be one of {', '.join(POOLING_MODES)}")
        self.pooling = pooling
        # Between stacked layers only; a single layer gets dropout on the readout below.
        self.lstm = nn.LSTM(feature_count, hidden_size, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        # Identity at 0 keeps the default model's random number use identical to the model without dropout.
        self.readout_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        readout_size = hidden_size * 2 if pooling == "mean_max" else hidden_size
        self.classifier = nn.Linear(readout_size, label_count)

    def forward(self, features, lengths):
        packed = pack_padded_sequence(features, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_outputs, (hidden, _) = self.lstm(packed)
        if self.pooling == "last":
            readout = hidden[-1]
        else:
            readout = self._pool_mean_max(packed_outputs)
        return self.classifier(self.readout_dropout(readout))

    @staticmethod
    def _pool_mean_max(packed_outputs):
        """Mean and max over each sequence's own time steps, so padding never enters either statistic."""
        outputs, output_lengths = pad_packed_sequence(packed_outputs, batch_first=True)
        output_lengths = output_lengths.to(outputs.device)
        time_steps = torch.arange(outputs.shape[1], device=outputs.device)
        valid = (time_steps[None, :] < output_lengths[:, None]).unsqueeze(-1)
        mean = (outputs * valid).sum(1) / output_lengths[:, None].to(outputs.dtype)
        maximum = outputs.masked_fill(~valid, float("-inf")).amax(1)
        return torch.cat([mean, maximum], dim=1)
