"""
Mnist tutorial main model
"""
import torch.nn as nn
import torch.Tensor as Tensor
from torch.utils.data import dataset
import torch.nn.functional as F
from ..weights_initializer import weights_init
from torch.nn import TransformerEncoder, TransformerEncoderLayer


PositionalEncoding

class TransformerModel(nn.Module):
    """ Define Transformer Class"""
    def __init__(self, ntoken:int, d_model:int, nhead: int, d_hid: int,
                nlayers: int, dropout: float=0.5):
        super().__init__()
        self.model_type = 'Transformer' """Model type"""
        self.pos_encoder = PositionalEncoding(d_model, dropout) """Add Positional Encoding"""
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.encoder = nn.Embedding(ntoken, d_model) # embedding ntoken into d-dimensional..
        self.d_model= d_model
        self.decoder = nn.Linear(d_model, ntoken)


class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int, dropout:float=0.1, max_len:int=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000,0)/d_model))
        pe= torch.zeros(max_len,1,d_model)
        pe[:,0,0::2] = torch.sin(position * div_term)
        pe[:,0,1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x:Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x+ self.pe[:x.size[0]]
        return self.dropout(x)
