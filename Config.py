import torch

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