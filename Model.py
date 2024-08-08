import torch
import torch.nn as nn
from Encoder import ViT_Encoder
from Decoder import GPT_Decoder
from Dataset import tokenizer
from Config import DEVICE

class ImageCaptionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_block = ViT_Encoder(config["enc_kwargs"])
        self.decoder_block = GPT_Decoder(config["dec_kwargs"])
        self.mapping_block = nn.Linear(config["enc_kwargs"]["embed_dim"], config["dec_kwargs"]["embed_dim"])
        
    def forward(self, img, tokens, attn_mask):
        x = self.encoder_block(img)
        img_encoding = self.mapping_block(x)
        x = self.decoder_block(tokens, img_encoding, attn_mask)
        
        return x;
    
    def generate(self, img_tensor, max_len):
        img_encoding = self.encoder_block(img_tensor)
        img_encoding = self.mapping_block(img_encoding)
        tokens = torch.tensor(data=[[tokenizer.get_vocab()['[BOS]']]], requires_grad=False).to(DEVICE)
        attn_mask = torch.tensor(data=[[1]], requires_grad=False).to(DEVICE)
        while tokens[0, -1]!=tokenizer.get_vocab()['[EOS]'] and tokens.shape[1]<max_len:
            logits = self.decoder_block(tokens, img_encoding, attn_mask, train=False)
            next_token = torch.argmax(logits[0, -1, :], dim=0).item()
            tokens = torch.cat((tokens, torch.tensor([[next_token]], requires_grad=False).to(DEVICE)), dim = -1).to(DEVICE)
            attn_mask = torch.cat((attn_mask, torch.tensor([[1]], requires_grad=False).to(DEVICE)), dim = -1).to(DEVICE)
        return tokens