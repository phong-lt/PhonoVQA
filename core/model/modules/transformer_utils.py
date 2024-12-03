import torch
from torch import nn, Tensor
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(SinusoidalPositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):

        return self.dropout(token_embedding + self.pos_embedding[:, :token_embedding.size(1)])

class TokenEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class BaseDecoder(nn.Module):
    def __init__(self, 
                emb_size: int,
                num_layers: int,
                n_head: int,
                batch_first: bool=True,
                ):
        super(BaseDecoder, self).__init__()
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=emb_size, nhead=n_head, batch_first = batch_first)
            ,num_layers=num_layers)
        
    def forward(self,
                tgt,
                memory,
                tgt_mask = None,
                memory_mask = None,
                tgt_key_padding_mask = None,
                memory_key_padding_mask = None,):

        return self.decoder(tgt=tgt,
                            memory=memory,
                            tgt_mask=tgt_mask,
                            memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,)
class BaseEncoder(nn.Module):
    def __init__(self, 
                emb_size: int,
                num_layers: int,
                n_head: int,
                batch_first: bool=True,
                ):
        super(BaseEncoder, self).__init__()
        
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=emb_size, nhead=n_head, batch_first = batch_first)
            ,num_layers=num_layers)
        
    def forward(self,
                src,
                src_mask = None,
                src_key_padding_mask = None,):

        return self.encoder(src=src,
                            mask=src_mask,
                            src_key_padding_mask=src_key_padding_mask,)
