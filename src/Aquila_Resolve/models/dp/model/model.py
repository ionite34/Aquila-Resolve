from abc import ABC, abstractmethod
from enum import Enum
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
from .utils import _make_len_mask, _generate_square_subsequent_mask, PositionalEncoding
from ..preprocessing.text import Preprocessor


class ModelType(Enum):
    TRANSFORMER = 'transformer'
    AUTOREG_TRANSFORMER = 'autoreg_transformer'

    def is_autoregressive(self) -> bool:
        """
        Returns: bool: Whether the model is autoregressive.
        """
        return self in {ModelType.AUTOREG_TRANSFORMER}  # pragma: no cover


class Model(torch.nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def generate(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates phonemes for a text batch

        Args:
          batch (Dict[str, torch.Tensor]): Dictionary containing 'text' (tokenized text tensor),
                       'text_len' (text length tensor),
                       'start_index' (phoneme start indices for AutoregressiveTransformer)

        Returns:
          Tuple[torch.Tensor, torch.Tensor]: The predictions. The first element is a tensor (phoneme tokens)
          and the second element  is a tensor (phoneme token probabilities)
        """
        pass  # pragma: no cover


class AutoregressiveTransformer(Model):

    def __init__(self,
                 encoder_vocab_size: int,
                 decoder_vocab_size: int,
                 end_index: int,
                 d_model=512,
                 d_fft=1024,
                 encoder_layers=4,
                 decoder_layers=4,
                 dropout=0.1,
                 heads=1):
        super().__init__()

        self.end_index = end_index
        self.d_model = d_model
        self.encoder = nn.Embedding(encoder_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.decoder = nn.Embedding(decoder_vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(d_model=d_model, nhead=heads, num_encoder_layers=encoder_layers,
                                          num_decoder_layers=decoder_layers, dim_feedforward=d_fft,
                                          dropout=dropout, activation='relu')
        self.fc_out = nn.Linear(d_model, decoder_vocab_size)

    @torch.jit.export
    def generate(self,
                 batch: Dict[str, torch.Tensor],
                 max_len: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inference pass on a batch of tokenized texts.

        Args:
          batch (Dict[str, torch.Tensor]): Dictionary containing the input to the model with entries 'text'
                                           and 'start_index'
          max_len (int): Max steps of the autoregressive inference loop.

        Returns:
          Tuple: Predictions. The first element is a Tensor of phoneme tokens and the second element
                 is a Tensor of phoneme token probabilities.
        """

        input = batch['text']
        start_index = batch['start_index']

        batch_size = input.size(0)
        input = input.transpose(0, 1)          # shape: [T, N]
        src_pad_mask = _make_len_mask(input).to(input.device)
        with torch.no_grad():
            input = self.encoder(input)
            input = self.pos_encoder(input)
            input = self.transformer.encoder(input,
                                             src_key_padding_mask=src_pad_mask)
            out_indices = start_index.unsqueeze(0)
            out_logits = []
            for i in range(max_len):
                tgt_mask = _generate_square_subsequent_mask(i + 1).to(input.device)
                output = self.decoder(out_indices)
                output = self.pos_decoder(output)
                output = self.transformer.decoder(output,
                                                  input,
                                                  memory_key_padding_mask=src_pad_mask,
                                                  tgt_mask=tgt_mask)
                output = self.fc_out(output)  # shape: [T, N, V]
                out_tokens = output.argmax(2)[-1:, :]
                out_logits.append(output[-1:, :, :])

                out_indices = torch.cat([out_indices, out_tokens], dim=0)
                stop_rows, _ = torch.max(out_indices == self.end_index, dim=0)
                if torch.sum(stop_rows) == batch_size:
                    break

        out_indices = out_indices.transpose(0, 1)  # out shape [N, T]
        out_logits = torch.cat(out_logits, dim=0).transpose(0, 1) # out shape [N, T, V]
        out_logits = out_logits.softmax(-1)
        out_probs = torch.ones((out_indices.size(0), out_indices.size(1)))
        for i in range(out_indices.size(0)):
            for j in range(0, out_indices.size(1)-1):
                out_probs[i, j+1] = out_logits[i, j].max()
        return out_indices, out_probs

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AutoregressiveTransformer':
        """
        Initializes an autoregressive Transformer model from a config.
        Args:
          config (dict): Configuration containing the hyperparams.

        Returns:
          AutoregressiveTransformer: Model object.
        """

        preprocessor = Preprocessor.from_config(config)
        return AutoregressiveTransformer(
            encoder_vocab_size=preprocessor.text_tokenizer.vocab_size,
            decoder_vocab_size=preprocessor.phoneme_tokenizer.vocab_size,
            end_index=preprocessor.phoneme_tokenizer.end_index,
            d_model=config['model']['d_model'],
            d_fft=config['model']['d_fft'],
            encoder_layers=config['model']['layers'],
            decoder_layers=config['model']['layers'],
            dropout=config['model']['dropout'],
            heads=config['model']['heads']
        )


def create_model(model_type: ModelType, config: Dict[str, Any]) -> Model:
    """
    Initializes a model from a config for a given model type.

    Args:
        model_type (ModelType): Type of model to be initialized.
        config (dict): Configuration containing hyperparams.

    Returns: Model: Model object.
    """
    if model_type is not ModelType.AUTOREG_TRANSFORMER:  # pragma: no cover
        raise ValueError(f'Unsupported model type: {model_type}. '
                         'Supported type: AUTOREG_TRANSFORMER')
    return AutoregressiveTransformer.from_config(config)


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Tuple[Model, Dict[str, Any]]:
    """
    Initializes a model from a checkpoint (.pt file).

    Args:
        checkpoint_path (str): Path to checkpoint file (.pt).
        device (str): Device to put the model to ('cpu' or 'cuda').

    Returns: Tuple: The first element is a Model (the loaded model)
             and the second element is a dictionary (config).
    """

    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_type = checkpoint['config']['model']['type']
    model_type = ModelType(model_type)
    model = create_model(model_type, config=checkpoint['config'])
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint
