import torch
import torch.nn as nn
import torch.nn.functional as F
from Dataset import tokenizer
from Config import config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

config['dec_kwargs']['vocab_size'] = tokenizer.vocab_size + 3

class GPTEmbedding(nn.Module):
    def __init__(self, context_length, embed_dim, vocab_size, dropout):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)
        self.position_embedding = nn.Parameter(torch.randn(size=(1, context_length, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        token_embeddings = self.token_embedding(x)
        embeddings = self.position_embedding[:, :x.shape[1], :] + token_embeddings
        output = self.dropout(embeddings)
        
        return output

class Masked_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv_proj = nn.Linear(embed_dim, embed_dim*3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, attn_mask):
        batch_size, con_length, embed_dim = x.shape
        qkv = self.qkv_proj(x).view(batch_size, con_length, 3, self.num_heads, embed_dim // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn_weights = torch.matmul(q, k.transpose(2, 3))
        if attn_mask is not None:
            # Ensure attn_mask has the shape [batch_size, 1, 1, seq_len] or [batch_size, num_heads, seq_len, seq_len]
            if attn_mask.dim() == 2:  # [batch_size, seq_len]
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            elif attn_mask.dim() == 3:  # [batch_size, 1, seq_len]
                attn_mask = attn_mask.unsqueeze(1)
        attn_mask = attn_mask.expand(batch_size, self.num_heads, con_length, con_length).bool()
        attn_weights = attn_weights.masked_fill(attn_mask, float("-inf"))
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights / ((embed_dim // self.num_heads)**0.5)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, con_length, embed_dim)
        
        output = self.out_proj(x)
        return output
    
class Cross_MHA(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x, img_encoding):
        batch_size, con_length, embed_dim_ = x.shape
        q = self.q_proj(x).view(batch_size, con_length, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(img_encoding).view(batch_size, 1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(img_encoding).view(batch_size, 1, self.num_heads, self.embed_dim // self.num_heads).permute(0, 2, 1, 3)
        attn_weights = torch.matmul(q, k.transpose(2, 3))
        attn_weights = attn_weights / ((self.embed_dim // self.num_heads)**0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, con_length, self.embed_dim)
        
        output = self.out_proj(attn_output)
        
        return output

class GPT_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.context_len = config["context_len"]
        self.num_decoders = config["num_decoders"]
        self.embedding_block = GPTEmbedding(config["context_len"], config["embed_dim"], config["vocab_size"], config["dropout"])
        self.masked_mha_block = Masked_MHA(config["embed_dim"], config["num_heads"])
        self.cross_mha_block = Cross_MHA(config["embed_dim"], config["num_heads"])
        self.layer_norm = nn.LayerNorm(normalized_shape=config["embed_dim"])
        self.mlp_block = nn.Sequential(
            nn.Linear(config["embed_dim"], config["embed_dim"]*4),
            nn.GELU(),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(config["embed_dim"]*4, config["embed_dim"]),
        )
        self.cls_head = nn.Linear(config["embed_dim"], config["vocab_size"])
    
    def custom_mask(self, attn_mask, context_len):
        mask = torch.triu(input=torch.ones(size=(context_len, context_len), requires_grad=False)*float('-inf'), diagonal=1).unsqueeze(0).repeat(attn_mask.shape[0], 1, 1)
        mask = mask.to(DEVICE)
        for i in range(mask.shape[0]):
            mask[i, attn_mask[i].logical_not(), :] = float("-inf")
            
        return mask
    
    def forward(self, tokens, img_encoding, attn_mask, train=True):
        if train == True:
            mask = self.custom_mask(attn_mask, self.context_len)
        else:
            mask = attn_mask
        for _ in range(self.num_decoders):
            x = self.embedding_block(tokens)
            residual = x
            x = self.masked_mha_block(x, mask)
            x = x + residual
            x = self.layer_norm(x)
            residual = x
            x = self.cross_mha_block(x, img_encoding)
            x = x + residual
            x = self.layer_norm(x)
            x = self.mlp_block(x)
        
        logits = self.cls_head(x)
        
        return logits