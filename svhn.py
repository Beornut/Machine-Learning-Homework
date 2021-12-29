import json
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils import data
from torchvision import transforms
from PIL import Image
import pandas as pd


class SVHN(nn.Module):
    def __init__(self):
        super(SVHN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.2)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=1),
            nn.Dropout(0.2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(192 * 7 * 5, 3072),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self.classifier_1 = nn.Linear(3072, 11)
        self.classifier_2 = nn.Linear(3072, 11)
        self.classifier_3 = nn.Linear(3072, 11)
        self.classifier_4 = nn.Linear(3072, 11)
        self.classifier_5 = nn.Linear(3072, 11)
        self.classifier_6 = nn.Linear(3072, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = x.view(x.size(0), 192 * 7 * 5)
        x = self.fc1(x)
        x = self.fc2(x)

        num_1 = self.classifier_1(x)
        num_2 = self.classifier_2(x)
        num_3 = self.classifier_3(x)
        num_4 = self.classifier_4(x)
        num_5 = self.classifier_5(x)
        num_6 = self.classifier_6(x)

        return num_1, num_2, num_3, num_4, num_5, num_6


class SVHNDataset(data.Dataset):
    def __init__(self, img_dir, label_path=None, label_len=None, transform=None):
        self.img_dir = img_dir
        self.img_path_list = os.listdir(img_dir)
        self.img_path_list.sort()
        if label_path:
            with open(label_path, 'r') as file:
                self.label_json = json.load(file)
                self.label_len = label_len
        else:
            self.label_json = None
            self.label_len = None
        self.transform = transform

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, i):
        img_filename = self.img_path_list[i]
        img = Image.open(os.path.join(self.img_dir, img_filename)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        if self.label_json and img_filename in self.label_json:
            label_list = self.label_json[img_filename]['label']
            label = [torch.tensor(label_list[j]) if j < len(label_list) else torch.tensor(10) for j in
                     range(self.label_len)]
        else:
            label = torch.zeros(32)
        return img, label, img_filename


def train(model, loss_func, optimizer, scheduler, train_loader, device, model_path):
    epoch_start = 50
    for i in range(epoch_start, epoch_start + 10):
        model.train()
        avg_loss = 0
        for j, (x, y, z) in enumerate(train_loader):
            x = x.to(device)
            y = [k.to(device) for k in y]
            optimizer.zero_grad()
            y_ = model(x)
            loss = loss_func(y_[0], y[0]) + loss_func(y_[1], y[1]) + loss_func(y_[2], y[2]) + loss_func(y_[3], y[3]) \
                   + loss_func(y_[4], y[4]) + loss_func(y_[5], y[5])
            loss.backward()
            optimizer.step()
            scheduler.step()
            avg_loss += loss.data / 6
            print('Epoch {}---Batch {}---loss: {}'.format(i, j, loss.data / 6))
        print('Epoch {}****Average loss: {}'.format(i, avg_loss / len(train_loader)))
    torch.save(model.state_dict(), model_path)


def val(model, val_loader, device):
    model.eval()
    accuracy = 0
    for x, y, z in val_loader:
        x = x.to(device)
        y = [k.to(device) for k in y]
        y_ = model(x)
        accuracy += torch.sum(torch.eq(torch.argmax(y_[0], dim=1), y[0]) &
                              torch.eq(torch.argmax(y_[1], dim=1), y[1]) &
                              torch.eq(torch.argmax(y_[2], dim=1), y[2]) &
                              torch.eq(torch.argmax(y_[3], dim=1), y[3]) &
                              torch.eq(torch.argmax(y_[4], dim=1), y[4]) &
                              torch.eq(torch.argmax(y_[5], dim=1), y[5]))
    print(accuracy / len(test_dataset))


def test(model, test_loader, device, result_path):
    model.eval()
    file_names = []
    file_codes = []
    l = 3
    for x, y, z in test_loader:
        x = x.to(device)
        y = [k.to(device) for k in y]
        y_ = model(x)
        file_names += z
        for i in range(x.size(0)):
            output = ''
            for j in y_:
                num = torch.argmax(j[i]).item()
                if num != 10:
                    output += str(num)
            file_codes.append(output)
    pd.DataFrame({'file_name': file_names, 'file_code': file_codes}).to_csv(result_path, index=False)


def check_label_len(path):
    with open(path, 'r') as file:
        label_json = json.load(file)
        max_len = 0
        max_len_file = None
        min_len = 10
        min_len_file = None
        for k, v in label_json.items():
            if max_len < len(v['label']):
                max_len = len(v['label'])
                max_len_file = k
            if min_len > len(v['label']):
                min_len = len(v['label'])
                min_len_file = k
        print(max_len_file, label_json[max_len_file]['label'])
        # train: 029929.png [1, 3, 5, 4, 5, 8]; val: 002998.png [0, 1, 2, 8, 7]
        print(min_len_file, label_json[min_len_file]['label'])
        # train: 000020.png [2]; val: 000000.png [5]


if __name__ == '__main__':
    run_in_colab = False
    if run_in_colab:
        dataset_path = '/content/'
        model_path = '/content/drive/MyDrive/model/svhn.checkpoint'
        result_path = '/content/drive/MyDrive/result/svnh.csv'
    else:
        dataset_path = './dataset'
        model_path = 'model/svhn.checkpoint'
        result_path = 'result/svnh.csv'
    train_img_dir = os.path.join(dataset_path, 'mchar', 'mchar_train')
    train_label_path = os.path.join(dataset_path, 'mchar', 'mchar_train.json')
    val_img_dir = os.path.join(dataset_path, 'mchar', 'mchar_val')
    val_label_path = os.path.join(dataset_path, 'mchar', 'mchar_val.json')
    test_img_dir = os.path.join(dataset_path, 'mchar', 'mchar_test')
    label_len = 6  # check_label_len(label_path)
    batch_size = 32
    train_dataset = SVHNDataset(train_img_dir, train_label_path, label_len, transform=transforms.Compose([
        transforms.Resize([80, 40]),
        transforms.RandomCrop([64, 32]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    val_dataset = SVHNDataset(val_img_dir, val_label_path, label_len, transform=transforms.Compose([
        transforms.Resize([80, 40]),
        transforms.CenterCrop([64, 32]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    test_dataset = SVHNDataset(test_img_dir, None, label_len, transform=transforms.Compose([
        transforms.Resize([80, 40]),
        transforms.CenterCrop([64, 32]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SVHN().to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=5e-5, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=625, gamma=0.9)
    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    train(model, loss_func, optimizer, scheduler, train_loader, device, model_path)
    val(model, val_loader, device)
    test(model, test_loader, device, result_path)


