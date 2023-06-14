import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from models.lstm_vae import LSTM_VAE
from models.embedding_model import GenreEmbedding_LSTM
from models.simple import SimpleLM
from datasets import Dataset, create_loaders
from datetime import datetime


def train_lstm_vae(model, train_loader, test_loader, epochs, lr=0.001):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        count = 0
        for x, genre_embedding in train_loader:
            optimizer.zero_grad()
            output, mu, logvar = model(x, genre_embedding)
            loss = model.vae_loss(output.transpose(1, 2), x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            count += 1
        print(f'Epoch {epoch}: {total_loss/count}')


def train_embedding_model(model, train_loader, val_loader, epochs, lr=0.001):
    print('Training embedding model...')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    test_embedding_model(model, val_loader)
    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            x, y = batch
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch}: {loss.item()}')
        test_embedding_model(model, val_loader)


# assume no batching -> default for batch_size = 1
def train_simple_model(dataset, epochs, lr=.001, seq_len=100):
    print('Training simple model...')
    data = torch.tensor(dataset.x) # shape: torch.Size([10000, 512])
    print(data.shape)
    model = SimpleLM(input_size=64, output_size=dataset.vocab_size, hidden_size=128) # NOTSURE: input_size=64??
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()
        for song in data:
            # each song is a torch tensor of shape 512
            data_ptr = 0    
            h = torch.zeros((1, model.hidden_size))
            c = torch.zeros((1, model.hidden_size))
            while data_ptr+seq_len+1<len(song):
                optimizer.zero_grad()
                input = song[data_ptr:data_ptr+seq_len]
                y_hat, (h, c) = model(torch.tensor(input), (h.detach(), c.detach()))
                y = song[data_ptr+1:data_ptr+seq_len+1]
                loss = criterion(y_hat.squeeze(), y)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                data_ptr += seq_len
        print(f'Epoch {epoch} loss per song: {epoch_loss/len(data)}')
    return model


def test_embedding_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch
            y_hat = model(x)
            predicted = torch.argmax(y_hat, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    print(f'Accuracy: {100 * correct / total}')


def save_model(model, model_name):
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    torch.save([model.kwargs, model.state_dict()],
               f'./models/{model_name}_{timestamp}.pt')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='lstm_vae', help='Model to train')
    parser.add_argument('--data_path', type=str, default='./data/data.csv', help='Path to data')
    parser.add_argument('--embedding_path', type=str, default='./models/genre_embeddings.pt', help='Path to embeddings (for VAE)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    if (args.model_type == 'lstm_vae'):
        dataset = Dataset(args.data_path)
        # Convert genre labels to embeddings
        embeds = torch.load('../models/genre_embeddings.p')
        new_y = np.zeros((dataset.y.shape[0], embeds.shape[1]), dtype=np.int64)
        for i in range(dataset.y.shape[0]):
            new_y[i] = embeds[dataset.y[i]]
        train_loader, test_loader = create_loaders(dataset.x, new_y, args.batch_size)
        model = LSTM_VAE(dataset.vocab_size, dataset.max_len, 64, 32, 50, 32)
        train_lstm_vae(model, train_loader, test_loader, args.epochs, args.lr)
        save_model(model, args.model_type)

    elif (args.model_type == 'embedding_model'):
        dataset = Dataset(args.data_path)
        train_loader, test_loader = create_loaders(dataset.x, dataset.y, args.batch_size)
        model = GenreEmbedding_LSTM(vocab_size=dataset.vocab_size, lstm_embedding_dim=64, lstm_hidden_dim=64, genre_embedding_dim=32, num_categories=dataset.num_labels)
        train_embedding_model(model, train_loader, test_loader, args.epochs, args.lr)
        torch.save(model.get_embeddings().detach(), './models/genre_embeddings.pt')
        save_model(model, args.model_type)

    elif (args.model_type == 'simple_model'):
        dataset = Dataset(args.data_path) # 10000, 512. 10k songs each of 512 words
        model = train_simple_model(dataset, args.epochs, args.lr)
        save_model(model, args.model_type)

    else:
        print('Invalid model type')


if __name__ == '__main__':
    main()
