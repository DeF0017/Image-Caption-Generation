import torch
import torch.nn as nn
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patch, dropout):
        super().__init__()
        self.patch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(start_dim=2)
        )
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embedding = nn.Parameter(torch.randn(size=(1, num_patch+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patch(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embedding + x
        x = self.dropout(x)
        return x
    

class ViT_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_encoders = config["num_encoders"],
        self.embedding_block = PatchEmbedding(config["in_channels"], config["patch_size"], config["embed_dim"], config["num_patch"], config["dropout"])
        self.layernorm = nn.LayerNorm(normalized_shape=config["embed_dim"])
        self.mha_block = nn.MultiheadAttention(
            embed_dim = config["embed_dim"],
            num_heads = config["num_heads"],
            batch_first = True,
            dropout = config["dropout"],
        )
        self.mlp_block = nn.Sequential(
            nn.Linear(config["embed_dim"], config["embed_dim"]*4),
            nn.GELU(),
            nn.Dropout(p=config["dropout"]),
            nn.Linear(config["embed_dim"]*4, config["embed_dim"]),
        )
    
    def forward(self, x):
        x = self.embedding_block(x)
        for _ in self.num_encoders:
            residual = x
            x = self.layernorm(x)
            x, _ = self.mha_block(x, x, x)
            x = x + residual
            
            residual = x
            x = self.layernorm(x)
            x = self.mlp_block(x) + residual
        
        return x[:, 0, :]