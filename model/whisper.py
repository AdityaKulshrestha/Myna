import math
import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, List


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding"""
    def __init__(self, dim, max_len=20000):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, dim, 2).float() * (torch.log(torch.tensor(10000.0)) / dim))  # Fixed exponentiation
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))                     # [T, D] -> [T, 1, D]; Unsqueeze at (1) index add additional dimension

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        seq_len = x.size(0)
        return self.pe[:seq_len]                                         # [:T, 1, D] For slicing
    

class MultiHeadAttention(nn.Module):
    """Custom implementation of multi-head attention"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisiable by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Q, K, V and output
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    
    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    
    def combine_heads(self, x):
        """Combine the heads back to original dimension"""
        batch_size, num_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    
    def forward(self, q, k, v, mask=None):
        """   
        Args:
            q, k, v: Query, Key, Value tensors [batch_size, seq_len, d_model]
            mask: Optional [batch_size, seq_len] or [batch_size, tgt_len, src_len]
                q_len = seq_len (Lenght of the input audio tensor)
                d_k/d_q/d_v = d_model / num_heads   (Embedding dimension divided into the num heads)
        Returns:
            Output tensors after attention [batch_size, seq_len, d_model]
        """
        batch_size = q.size(0)
        # Linear projections
        q = self.q_linear(q)                # [batch_size, q_len, d_model]
        k = self.k_linear(k)                # [batch_size, q_len, d_model]
        v = self.v_linear(v)                # [batch_size, q_len, d_model]

        # Split heads
        q = self.split_heads(q)             # [batch_size, num_heads, q_len, d_k]
        k = self.split_heads(k)             # [batch_size, num_heads, k_len, d_k]
        v = self.split_heads(v)             # [batch_size, num_heads, v_len, d_k]

        # Scaled dot-product attention
        # Transpose k for matrix multiplication [batch_size, num_heads, d_k, k_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # Apply mask if provided
        if mask is not None:
            # For self-attention: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len] 
            # For encoder-decoder attention: [batch_size, tgt_len, src_len] -> [batch_size, 1, tgt_len, src_len]
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

            scores = scores.masked_fill(mask == 1, -1e9)                # Applies -1e9 where the value is 0 (for encoder -> Padding)

        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        context = torch.matmul(attention_weights, v)                # [batch_size, num_heads, q_len, d_k]

        # Combine heads
        context = self.combine_heads(context)                       # [batch_size, q_len, d_model]

        # Final output linear projection
        output = self.out_linear(context)                           # [batch_size, q_len, d_model]                                  
        return output, attention_weights
    

class FeedForward(nn.Module):
    """Simple feed-forward network with ReLU activation"""
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        return self.linear2(F.relu(self.linear1(x)))


class EncoderBlock(nn.Module):
    """Single transformer encoder block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)


        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional mask [batch_size, seq_len]

        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self attention with residual connection and layer norm
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual connection and layer nomr
        ffn_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class DecoderBlock(nn.Module):
    """Single transformer decoder block"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)

        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        """
        Args:
            x: Target tensor [batch_size, tgt_len, d_model]
            enc_output: Encoder output [batch_size, src_len, d_model]
            src_mask: Source mask for cross-attention [batch_size, src_len]
            tgt_mask: Target mask for self-attention [batch_size, tgt_len] or causal mask [batch_size, tgt_len, tgt_len]

        Returns:
            Output tensor [batch_size, tgt_len, d_model]
        """
        # Self attention with residual connection and layer norm
        self_attn_output, _ = self.self_attention(x, x, x, tgt_mask)
        self.norm1(x + self.dropout(self_attn_output))

        # Cross attention with encoder output
        cross_attn_output, _  = self.cross_attention(x, enc_output, enc_output, src_mask)       # Need to modify src_mask for padding_masking
        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed forward with residual connection and layer norm
        ffn_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ffn_output))

        return x


class WhisperEncoder(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_layers: int, num_heads: int, max_len: int):
        super(WhisperEncoder, self).__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.conv1 = nn.Conv1d(input_dim, embed_dim, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1)

        self.pos_emb = SinusoidalPositionalEmbedding(embed_dim, max_len)

        self.transformer = EncoderBlock(d_model=embed_dim, num_heads=num_heads, d_ff=4*embed_dim, dropout=0.1)
        # self.transformer = nn.ModuleList([
        #     EncoderBlock(
        #         d_model=embed_dim,
        #         num_heads=num_heads,
        #         d_ff=4*embed_dim,
        #         dropout=0.1
        #     )
        #     for _ in range(num_layers)
        # ])

    def forward(self, x):
        """
        Args:
            x: Mel spectogram [batch, features, time]
        """
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        x = F.gelu(x)

        # Transpose: [batch, features, time] -> [batch, time, features]
        x = x.transpose(1, 2)                       # [B, T, F]

        # Generate padding mask based on zero values in the spectrogram
        # We assume that padded regions in mel spectrogram are zeros
        padding_mask = (x.abs().sum(dim=-1) == 0)

        # Linear projection
        # x = self.linear(x)

        # Transpose for transformer: [batch, time, dim] -> [time, batch, dim]
        x = x.transpose(0, 1)                                           # [T, B, D]

        # Add positional encoding
        x = x + self.pos_emb(x)                                         # [T, B, D]

        # Apply transformer with padding mask
        x = x.transpose(0, 1)
        x = self.transformer(x, mask=padding_mask)      # [T, B, D]

        # # Transpose back: [time, batch, dim] -> [batch, time, dim]
        # x = x.transpose(0, 1)                                            # [T, B, D]

        return x, padding_mask
    

class WhisperDecoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_len: int, pad_token_id: int = 0):
        super(WhisperDecoder, self).__init__()
        # Save padding token ID for mask generation
        self.pad_token_id = pad_token_id 

        # Token embedding 
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        # Positional embedding (sinusoidal)
        self.pos_emb = SinusoidalPositionalEmbedding(embed_dim, max_len)

        self.transformer = DecoderBlock(d_model=embed_dim, num_heads=num_heads, d_ff=4*embed_dim, dropout=0.1)
        # self.transformer = nn.ModuleList([
        #     DecoderBlock(
        #         d_model=embed_dim,
        #         num_heads=num_heads,
        #         d_ff=4*embed_dim,
        #         dropout=0.1
        #     )
        #     for _ in range(num_layers)
        # ])
        # Final projection to vocab
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens, memory, memory_padding_mask=None):
        """
        Args:
            tokens: Token indices [batch, seq_len]
            memory: Encoder output [batch, source_len, dim] 
            memory_padding_mask: Padding mask for memory [batch, source_len]
        """
        # Generate padding mask for tokens based on pad_token_id
        tokens_padding_mask = tokens == self.pad_token_id                              # [B, S]
        # Embed tokens
        x = self.token_embedding(tokens)                                                    # [B, S, D]

        # Transpose for transformer: [batch, seq, dim] -> [seq, batch, dim]
        x = x.transpose(0, 1)                                                               # [S, B, D]

        # Add positional encoding
        x = x + self.pos_emb(x)                                                             # [S, B, D]
        x = x.transpose(0, 1)                                                               # [B, S, D]

        # Create causal mask for autoregressive decoding
        tgt_len = tokens.size(1)
       # [B, S, S] Generates a mask of causal masking with end values as -inf
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=x.device), diagonal=1).bool().unsqueeze(0)            # Because we are already doing this in the attention block -> False -> 1e-9

        # Apply transformer decoder
        x = self.transformer(                                                               # [S, B, D]
            x, memory,
            tgt_mask=tgt_mask,
            src_mask=memory_padding_mask,
        )                                                         # [B, S, D]

        # Projects to vocabulary
        logits = self.fc(x)                                                                  # [B, S, V]

        return logits, tokens_padding_mask       
    

class Whisper(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        encoder_input_dim: int = 80,        # Mel bins
        embed_dim: int = 512,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        max_source_len: int = 3000,
        max_target_len: int = 448,
        pad_token_id: int = 0,
    ):
        super(Whisper, self).__init__()
    
        self.encoder = WhisperEncoder(
            input_dim=encoder_input_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            max_len=max_source_len,
        )

        self.decoder = WhisperDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            max_len=max_target_len,
            pad_token_id=pad_token_id
        )

        self.pad_token_id = pad_token_id

    def forward(
        self,
        mel: torch.Tensor,
        tokens: torch.Tensor
    ):
        """
        Forward pass through the model.

        Args:
            mel: Log-Mel spectrogram [batch, features, time]
            tokens: Target token indices [batch, seq_len]

        Returns:
            logits: Output logits [batch, seq_len, vocab_size]
            padding_masks: Dictionary of padding masks
        """
        # Encoder forward pass - generate padding mask from mel spectrogram
        encoder_out, mel_padding_mask = self.encoder(mel)                       # [B, T, D], [B, T]
        # Decoder forward pass - generates padding mask from tokens
        logits, tokens_padding_mask = self.decoder(                             # [B, S, V], [B, S]
            tokens,
            encoder_out,
            mel_padding_mask
        )

        return logits, {
            "mel_padding_mask": mel_padding_mask,
            "tokens_padding_mask": tokens_padding_mask
        }
    
    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        max_len: int = 448,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
    ):
        """
        Autoregressive generation of tokens

        Args:
            mel: Log-Mel spectrogram [batch, features, time]
            max_len: Maximum length of generated sequence
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep for top-k sampling
            top_p: Probability threshold for nucleus sampling
            bos_token_id: Beginning of sequence token id
            eos_token_id: End of sequence token id

        Returns:
            generated_ids: Generated token Ids [batch, seq_len]
        """
        batch_size = mel.size(0)
        device = mel.device 

        # Encode the mel spectrogram
        encoder_output, _ = self.encoder(mel)                               # [B, T, D], [B, T]

        # Initialize with BOS Token
        input_ids = torch.full(
            (batch_size, 1),
            bos_token_id,
            dtype=torch.long,
            device=device
        )

        # Generate until max_length or all sequences have EOS
        for _ in range(max_len - 1):
            # Get logits for next token
            logits, _ = self.decoder(input_ids, encoder_output)                 # [B, S, V]
            next_token_logits = logits[: -1, :]                                 # [B, V]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('Inf')

            # Apply top-k (nucleus) sampling
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshhold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above the threshold        
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted indices to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1,
                    index=sorted_indices,
                )
                next_token_logits[indices_to_remove] = float('Inf')

            # Samplie next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)                            # [B, 1]

            # Concatenate new token with previous tokens
            input_ids = torch.cat([input_ids, next_token], dim=-1)                          # [B, S+1]

            # Check if all sequences have EOS
            eos_mask = (next_token.squeeze(-1) == eos_token_id)
            if eos_mask.all():
                break

        return input_ids