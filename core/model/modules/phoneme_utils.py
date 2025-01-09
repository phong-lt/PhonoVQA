import json
import torch
from torch import nn
import math
from .transformer_utils import SinusoidalPositionalEncoding

class PhonemeEmbedding(nn.Module):
    def __init__(self, 
                onset_vocab_size,
                rhyme_vocab_size,
                tone_vocab_size,
                onset_embed_dim=256,
                rhyme_tone_embed_dim=256,):
        super(PhonemeEmbedding, self).__init__()
        

        self.onset_embed_dim = onset_embed_dim
        self.rhyme_tone_embed_dim = rhyme_tone_embed_dim

        self.onset_embedding = nn.Embedding(onset_vocab_size, onset_embed_dim)
        self.rhyme_embedding = nn.Embedding(rhyme_vocab_size, rhyme_tone_embed_dim)
        self.tone_embedding = nn.Embedding(tone_vocab_size, rhyme_tone_embed_dim)

        self.onset_positional_encoding = SinusoidalPositionalEncoding(
            onset_embed_dim, dropout=0.1)
        
        self.rhyme_positional_encoding = SinusoidalPositionalEncoding(
            rhyme_tone_embed_dim, dropout=0.1)

        self.tone_positional_encoding = SinusoidalPositionalEncoding(
            rhyme_tone_embed_dim, dropout=0.1)
      

    def forward(self, phoneme_tensor):
        onset_indices = phoneme_tensor[:, :, 0]
        rhyme_indices = phoneme_tensor[:, :, 1]
        tone_indices = phoneme_tensor[:, :, 2]

        onset_emb = self.onset_positional_encoding(self.onset_embedding(onset_indices) * math.sqrt(self.onset_embed_dim))
        rhyme_emb = self.rhyme_positional_encoding(self.rhyme_embedding(rhyme_indices) * math.sqrt(self.rhyme_tone_embed_dim))
        tone_emb = self.tone_positional_encoding(self.tone_embedding(tone_indices) * math.sqrt(self.rhyme_tone_embed_dim))

        word_embeddings = torch.cat((onset_emb, rhyme_emb, tone_emb), dim=-1)

        return word_embeddings
    
class PhonemeEmbedding_v2(nn.Module):
    def __init__(self, 
                onset_vocab_size,
                rhyme_vocab_size,
                tone_vocab_size,
                onset_embed_dim=256,
                rhyme_tone_embed_dim=256,):
        super(PhonemeEmbedding_v2, self).__init__()
        

        self.onset_embed_dim = onset_embed_dim
        self.rhyme_tone_embed_dim = rhyme_tone_embed_dim

        self.onset_embedding = nn.Embedding(onset_vocab_size, onset_embed_dim)
        self.rhyme_embedding = nn.Embedding(rhyme_vocab_size, rhyme_tone_embed_dim)
        self.tone_embedding = nn.Embedding(tone_vocab_size, rhyme_tone_embed_dim)
      

    def forward(self, phoneme_tensor):
        onset_indices = phoneme_tensor[:, :, 0]
        rhyme_indices = phoneme_tensor[:, :, 1]
        tone_indices = phoneme_tensor[:, :, 2]

        onset_mask = onset_indices == 1
        rhyme_mask = rhyme_indices == 1
        tone_mask = tone_indices == 1 

        onset_emb = self.onset_embedding(onset_indices) * math.sqrt(self.onset_embed_dim)
        rhyme_emb = self.rhyme_embedding(rhyme_indices) * math.sqrt(self.rhyme_tone_embed_dim)
        tone_emb = self.tone_embedding(tone_indices) * math.sqrt(self.rhyme_tone_embed_dim)



        return onset_emb, rhyme_emb, tone_emb, onset_mask, rhyme_mask, tone_mask
