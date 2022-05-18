import math
import torch


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout=0.1, max_len=5000) -> None:
        """
        Initializes positional encoding.

        Args:
            d_model (int): Dimension of model.
            dropout (float): Dropout after positional encoding.
            max_len: Max length of precalculated position sequence.
        """

        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.scale = torch.nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # shape: [T, N]
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)


def _generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    mask = torch.triu(torch.ones(sz, sz), 1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


def _make_len_mask(inp: torch.Tensor) -> torch.Tensor:
    return (inp == 0).transpose(0, 1)


def _get_len_util_stop(sequence: torch.Tensor, end_index: int) -> int:
    for i, val in enumerate(sequence):
        if val == end_index:
            return i + 1
    return len(sequence)  # pragma: no cover
