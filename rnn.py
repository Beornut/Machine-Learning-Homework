import os
import re
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


class RNN(nn.Module):
    def __init__(self, sentence_len, dictionary_len, embedding_len, hidden_size=128, layer_num=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(dictionary_len, embedding_len)
        self.lstm = nn.LSTM(embedding_len, hidden_size, num_layers=layer_num, batch_first=True, bidirectional=True)
        self.pool = nn.AvgPool2d((sentence_len, 1))
        self.actv = nn.ReLU()
        self.dp = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.logistic = nn.Sigmoid()

    def forward(self, text):
        x = self.embedding(text)
        x, _ = self.lstm(x)
        x = self.dp(x)
        x = self.actv(self.fc1(x))
        x = self.actv(self.pool(x).squeeze())
        x = self.fc2(x)
        return self.logistic(x).squeeze()


class IMDBDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x)
        self.y = torch.tensor(y)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def build_dictionary(path, pkl_path, min_freq):
    file_path = [path + '/imdb/train/neg', path + '/imdb/train/pos', path + '/imdb/test/neg', path + '/imdb/test/pos']
    all_word = {}
    dictionary = {}
    id = 2
    max_sentence_len = 0
    for i in file_path:
        file_list = os.listdir(i)
        for j in file_list:
            with open(i + '/' + j, 'r') as file:
                sentence = file.read()
                lower = sentence2words(sentence)
                for k in lower:
                    if k not in all_word:
                        all_word[k] = 1
                    else:
                        all_word[k] += 1
                if len(lower) > max_sentence_len:
                    max_sentence_len = len(lower)
    for k in all_word.keys():
        if all_word[k] > min_freq:
            dictionary[k] = id
            id += 1
    with open(pkl_path, 'wb') as file:
        pickle.dump(dictionary, file)
    return max_sentence_len


def sentence2words(sentence):
    s = re.sub('<br\\s*/>', ' ', sentence)
    s = re.sub('([.,;:!?()/\'\"\\-*]+)', ' ', s)
    s = re.sub('\\s{2,}', ' ', s)
    split = s.split()
    return [w.lower() for w in split]


def build_dataset(dataset_path, dict_pkl_path, data_pkl_path, sentence_len):
    x = []
    y = []
    with open(dict_pkl_path, 'rb') as file:
        dictionary = pickle.load(file)
    if dictionary:
        file_path = [dataset_path + '/imdb/train/neg', dataset_path + '/imdb/train/pos',
                     dataset_path + '/imdb/test/neg', dataset_path + '/imdb/test/pos']
        for i in file_path:
            file_list = os.listdir(i)
            for j in file_list:
                with open(i + '/' + j, 'r') as file:
                    sentence = file.read()
                    lower = sentence2words(sentence)
                    vector = []
                    for k in lower:
                        if k in dictionary:
                            vector.append(dictionary[k])
                        else:
                            vector.append(1)
                        if len(vector) >= sentence_len:
                            break
                    while len(vector) < sentence_len:
                        vector.append(0)
                    x.append(vector)
                    if 'pos' in i:
                        y.append(1)
                    else:
                        y.append(0)
        with open(data_pkl_path, 'wb') as file:
            pickle.dump({'x': x, 'y': y}, file)


if __name__ == '__main__':
    run_in_colab = False
    min_freq = 1000
    sentence_len = 128
    embedding_len = 32
    batch_size = 100
    if run_in_colab:
        dataset_path = '/content/drive/MyDrive/dataset'
        model_path = '/content/drive/MyDrive/model/rnn.checkpoint'
    else:
        dataset_path = './dataset'
        model_path = 'model/rnn.checkpoint'
    dict_pkl_path = dataset_path + '/imdb/dictionary.pkl'
    data_pkl_path = dataset_path + '/imdb/dataset.pkl'
    if not os.path.exists(dict_pkl_path):
        build_dictionary(dataset_path, dict_pkl_path, min_freq)
        build_dataset(dataset_path, dict_pkl_path, data_pkl_path, sentence_len)
    with open(dict_pkl_path, 'rb') as file:
        dictionary = pickle.load(file)
        vector_size = len(dictionary) + 2
    with open(data_pkl_path, 'rb') as file:
        data = pickle.load(file)
    train_dataset = IMDBDataset(data['x'][:25000], data['y'][:25000])
    test_dataset = IMDBDataset(data['x'][25000:], data['y'][25000:])
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True)
    model = RNN(sentence_len, vector_size, embedding_len)
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
    else:
        loss_func = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        batch_num = 0
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            y_ = model(x)
            loss = loss_func(y_.to(torch.float), y.to(torch.float))
            loss.backward()
            optimizer.step()
            print('Train batch:{}, loss:{}'.format(batch_num, loss.data))
            batch_num += 1
        torch.save(model.state_dict(), model_path)

    accuracy = 0
    for x, y in test_loader:
        model.eval()
        y_ = (model(x) > 0.5)
        accuracy += torch.sum(torch.eq(y_, y))
    print(accuracy / 25000)
