import torch
from PIL import Image
from torchvision import transforms
import transformers
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.add_special_tokens({"bos_token": "[BOS]",
               "eos_token": "[EOS]",
               "pad_token": "[PAD]"}) 

class ImageCaptionDataset(Dataset):
    def __init__(self, dataframe, img_size, context_len):
        self.df = dataframe
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.context_len = context_len
        self.df_len = dataframe.shape[0]
        
    def __len__(self):
        return self.df_len
    
    def __getitem__(self, idx):
        img_path = '/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/' + self.df.iloc[idx, 0]
        img = Image.open(img_path)
        comment = self.df.iloc[idx, 1]
        img_tensor = self.transforms(img)
        tokens = tokenizer(comment, max_length=self.context_len+1, padding='max_length', truncation=True, return_tensors='pt')
        
        return img_tensor, tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze()