import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tokenizers import Tokenizer
from lyricsgenius import Genius
genius = Genius("XN6f6ECwHaLNf0dun2Sobkx6bp5ZwoIllko41uM-2qb7ONKqGpx4rKunAsVwtvcS",
                timeout=100, skip_non_songs=True)


class SongTokenizer:
    def __init__(self, max_len=512):
        self.tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        self.tokenizer.enable_padding(
            pad_id=0, pad_token="[PAD]", length=max_len)
        self.tokenizer.enable_truncation(max_length=max_len)
        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode_text(self, text):
        return self.tokenizer.encode(text).ids

    def decode_text(self, ids):
        return self.tokenizer.decode(ids)


class Dataset:
    def __init__(self, data_file, max_len=512):
        self.max_len = max_len
        self.labels = ['pop female', 'pop male',
                       'rock female', 'rock male',
                       'rap female', 'rap male',
                       'country female', 'country male',
                       'rb female', 'rb male']
        self.num_labels = len(self.labels)
        self.labels_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_labels = {i: label for i, label in enumerate(self.labels)}

        self.tokenizer = SongTokenizer(max_len)
        self.vocab_size = self.tokenizer.vocab_size

        self.x, self.y = self.process(pd.read_csv(data_file))
        self.size = len(self.x)
        print(f'Dataset of {self.size} songs loaded...')

    def __len__(self):
        return self.size

    def process(self, df):
        x = np.zeros((len(df), self.max_len), dtype=np.int64)
        y = np.zeros((len(df)), dtype=np.int64)
        for i in range(len(df)):
            row = df.iloc[i]
            x[i, :] = self.tokenizer.encode_text(row['lyrics'])
            y[i] = self.labels_to_idx[row['genre'] + ' ' + row['gender']]
        return x, y


class Songs(Dataset):
    # len(lyrics) = len(genre_gender) = number of songs
    def __init__(self, lyrics, genre_gender):
        self.lyrics = torch.from_numpy(lyrics)
        self.genre_gender = torch.from_numpy(genre_gender)

    def __getitem__(self, index):
        lyric = self.lyrics[index]
        label = self.genre_gender[index]
        return lyric, label

    def __len__(self):
        return len(self.lyrics)

# data_x: lyrics
# data_y: genre+gender label (n genres: data_y values will be [0, 2n-1])


def create_loaders(data_x, data_y, batch_size):
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
    train_loader = torch.utils.data.DataLoader(Songs(train_x, train_y), batch_size)
    test_loader = torch.utils.data.DataLoader(Songs(test_x, test_y), batch_size)
    print('Data loaders created...')
    print('Batch size: {}'.format(batch_size))
    print('Train size: {}'.format(len(train_x)))
    print('Test size: {}'.format(len(test_x)))
    return train_loader, test_loader


def load_dataset_lyricgenius(save_path):
    data = {'genre': [], 'gender': [], 'lyrics': []}
    pop_female_artists = ['Adele', 'Ariana Grande', 'Beyoncé', 'Billie Eilish', 'Britney Spears', 'Christina Aguilera', 'Dua Lipa', 'Halsey', 'Jennifer Lopez',
                          'Katy Perry', 'Lady Gaga', 'Demi Lovato', 'Pussycat Dolls', 'Miley Cyrus', 'P!nk', 'Rihanna', 'Selena Gomez', 'Shakira', 'Kelly Clarkson', 'Gwen Stefani']
    for female_artist in pop_female_artists:
        try:
            artist = genius.search_artist(female_artist, max_songs=1)
        except:
            print('Error searching {}'.format(female_artist))
            continue
        songs = artist.songs
        for song in songs:
            data['lyrics'].append(song.lyrics)
            data['genre'].append('pop')
            data['gender'].append('female')
            print('Added {} to dataset'.format(song.full_title))

    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    return df


# load_dataset_lyricgenius('./data/data.csv')
# dataset = Dataset('./data/data.csv')
# train_loader, test_loader = create_loaders(dataset.x, dataset.y, 2)
# for batch in train_loader:
#     x, y = batch
#     print(x.shape)
#     print(x)
#     print(y.shape)
