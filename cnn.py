import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5, stride=1, padding=2)
        self.actv1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, 3, stride=1, padding=1)
        self.actv2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.flt = nn.Flatten(1)
        self.fc = nn.Linear(8 * 7 * 7, 10)

    def forward(self, img):
        x = self.conv1(img)
        x = self.actv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.actv2(x)
        x = self.pool2(x)
        x = self.flt(x)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    run_in_colab = False
    if run_in_colab:
        dataset_path = '/content/drive/MyDrive/dataset'
        model_path = '/content/drive/MyDrive/model/cnn.checkpoint'
    else:
        dataset_path = './dataset'
        model_path = 'model/cnn.checkpoint'
    batch_size = 32
    train_dataset = MNIST(dataset_path, train=True, transform=transforms.ToTensor())
    test_dataset = MNIST(dataset_path, train=False, transform=transforms.ToTensor())
    test_size = len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, test_size, shuffle=False)
    model = CNN()
    if os.path.isfile(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
    else:
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        batch_num = 0
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            y_ = model(x)
            loss = loss_func(y_, y)
            loss.backward()
            optimizer.step()
            batch_num += 1
            print('Train batch:{}, loss:{}'.format(batch_num, loss.data))
        torch.save(model.state_dict(), model_path)

    accuracy = 0
    for x, y in test_loader:
        model.eval()
        y_ = torch.argmax(model(x), dim=1)
        accuracy += torch.sum(torch.eq(y_, y))
    print(accuracy / test_size)

