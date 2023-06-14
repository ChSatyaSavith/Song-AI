import torch
from src.datasets import SongTokenizer
from src.models.lstm_vae import LSTM_VAE
from src.models.embedding_model import GenreEmbedding_LSTM
from src.models.simple import SimpleLM

# SimpleLM
tokenizer = SongTokenizer()
kwargs, state = torch.load('./models/simple_model_2023-04-24_15-20-08.pt')
model = SimpleLM(**kwargs)
model.load_state_dict(state)
model.eval()
song = model.sample(100)
print("generated:", song)
print(tokenizer.decode_text(song.tolist()))