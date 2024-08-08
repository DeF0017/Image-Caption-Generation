import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Model import ImageCaptionModel
from Dataset import ImageCaptionDataset, tokenizer
from Config import config

IMG_SIZE = 224
CONTEXT_LENGTH = 256
BATCH_SIZE = 4
PATCH_SIZE = 16
IN_CHANNELS = 3
NUM_HEADS_ENC = 8
NUM_ENCODERS = 4
NUM_PATCH = (IMG_SIZE // PATCH_SIZE)**2
EMBED_DIM_ENC = (PATCH_SIZE**2)*IN_CHANNELS
NUM_HEADS_DEC = 8
EMBED_DIM_DEC = 512
NUM_DECODERS = 4
DROPOUT=0.001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR = 2e-6
EPOCHS = 10
SAVE_MODEL = True
LOAD_MODEL = True
FILENAME = "ImageGen.pth.tar"

encoder_kwargs = {
    "num_encoders" : NUM_ENCODERS,
    "num_heads": NUM_HEADS_ENC,
    "num_patch": NUM_PATCH,
    "patch_size": PATCH_SIZE,
    "in_channels": IN_CHANNELS,
    "embed_dim": EMBED_DIM_ENC,
    "pretrained_model_name": None,
    "device": DEVICE,
    "dropout": DROPOUT
}
decoder_kwargs = {
    "embed_dim": EMBED_DIM_DEC,
    "context_len": CONTEXT_LENGTH,
    "num_decoders": NUM_DECODERS,
    "num_heads": NUM_HEADS_DEC,
    "device": DEVICE,
    "dropout": DROPOUT
# should add ignore_index and vocab_size before sending to the model
}
config = {
    "enc_kwargs": encoder_kwargs,
    "dec_kwargs": decoder_kwargs,
    "device": DEVICE
}

df = pd.read_csv('/kaggle/input/flickr-image-dataset/flickr30k_images/results.csv', delimiter='|')
df.drop(columns=[' comment_number'], axis=1, inplace=True)
df.drop_duplicates(subset=['image_name'], inplace=True)
df[' comment'] = '[BOS] ' + df[' comment'] + ' [EOS]'
df.reset_index(drop=True, inplace=True)

train_size = int(len(df)*0.90)
test_size = len(df)-train_size
train_data, test_data = train_test_split(df, test_size=0.1, random_state=42)
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

def save_checkpoint(model, optimizer, epoch, loss, filename='checkpoint.pth'):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, lr, filename='checkpoint.pth'):
    checkpoint = torch.load(filename, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print(f"Checkpoint loaded from {filename}, epoch {epoch}, loss {loss}")
    return epoch, loss

def train(model, optim, loader, tokenizer):
    loop = tqdm(loader, leave=True)
    total_loss = 0
    for img_tensor, tokens, attn_mask in loop:
        input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]
        attn_mask = attn_mask[:, :-1]
        #print(attn_mask.shape)
        img_tensor = img_tensor.to(DEVICE)
        input_tokens = input_tokens.to(DEVICE)
        targets_tokens = target_tokens.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
        
        logits = model(img_tensor, input_tokens, attn_mask)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets_tokens.reshape(-1), ignore_index=tokenizer.get_vocab()[tokenizer.pad_token])
            
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        total_loss += loss.item()
        loop.set_description(f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    return avg_loss

def eval_(model, loader, tokenizer):
    loop = tqdm(loader, leave=True)
    total_loss = 0
    for img_tensor, tokens, attn_mask in loop:
        input_tokens, target_tokens = tokens[:, :-1], tokens[:, 1:]
        attn_mask = attn_mask[:, :-1]
        img_tensor = img_tensor.to(DEVICE)
        input_tokens = input_tokens.to(DEVICE)
        targets_tokens = target_tokens.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)
        
        logits = model(img_tensor, input_tokens, attn_mask)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets_tokens.reshape(-1), ignore_index=tokenizer.get_vocab()[tokenizer.pad_token])
        
        total_loss += loss.item()
        #loop.set_description(f"Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(loader)
    return avg_loss


model = ImageCaptionModel(config).to(DEVICE)
def main():
    optimizer = optim.AdamW(list(model.parameters()), lr=LR, weight_decay=1e-5)
    
    train_dataset = ImageCaptionDataset(train_data, img_size=IMG_SIZE, context_len=CONTEXT_LENGTH)
    test_dataset = ImageCaptionDataset(test_data, img_size=IMG_SIZE, context_len=CONTEXT_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    if LOAD_MODEL:
        load_checkpoint(model, optimizer, LR, "/kaggle/input/imagegen-checkpoint-l3/ImageGen.pth (9).tar")
    for num in range(EPOCHS):
        print("Epoch:", num+1)
        train_loss = train(model, optimizer, train_loader, tokenizer)
        eval_loss = eval_(model, test_loader, tokenizer)
        print("Train Loss: ", train_loss)
        print("Eval Loss: ", eval_loss)

        if (num+1)%5==0 and SAVE_MODEL:
            save_checkpoint(model, optimizer, num, train_loss, FILENAME)

if __name__ == "__main__":
    main()

trnsfrms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
def inference(img_path, model):
    img = Image.open(img_path)
    img_tensor = trnsfrms(img)
    img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(DEVICE)
    final_tokens = model.generate(img_tensor, 256)
    final_tokens = final_tokens.view(-1)
    final_tokens = list(final_tokens)
    return tokenizer.decode(token_ids=[token.item() for token in final_tokens])

img_path = '/kaggle/input/flickr-image-dataset/flickr30k_images/flickr30k_images/100652400.jpg'
img = Image.open(img_path)
plt.imshow(img)
print(inference(img_path, model))