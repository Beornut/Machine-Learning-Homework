import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
from torchvision import transforms


class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        cfg = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000)
        )


class FashionVGG16(nn.Module):
    def __init__(self, vgg16, path, pretrained=False):
        super().__init__()
        if pretrained:
            vgg16.load_state_dict(torch.load(path))
        self.features = vgg16.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    run_in_colab = False
    if run_in_colab:
        dataset_path = '/content/drive/MyDrive/dataset'
        model_path = '/content/drive/MyDrive/model/vgg.checkpoint'
        vgg16_path = '/content/drive/MyDrive/model/vgg16.pth'
    else:
        dataset_path = './dataset'
        model_path = 'model/vgg.checkpoint'
        vgg16_path = 'model/vgg16.pth'
    batch_size = 100
    train_dataset = FashionMNIST(dataset_path, train=True, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                                         transforms.ToTensor()]))
    test_dataset = FashionMNIST(dataset_path, train=False, transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                                         transforms.ToTensor()]))
    test_size = len(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    vgg16 = VGG16()
    if os.path.isfile(model_path):
        model = FashionVGG16(vgg16, vgg16_path)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
    else:
        model = FashionVGG16(vgg16, vgg16_path, True)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        batch_num = 0
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            x = x.repeat(1, 3, 1, 1)
            y_ = model(x)
            loss = loss_func(y_, y)
            loss.backward()
            optimizer.step()
            batch_num += 1
            print('Train batch:{}, loss:{}'.format(batch_num, loss.data))
            if batch_num >= 10:
                break
        torch.save(model.state_dict(), model_path)

    accuracy = 0
    batch_num = 0
    for x, y in test_loader:
        model.eval()
        x = x.repeat(1, 3, 1, 1)
        y_ = torch.argmax(model(x), dim=1)
        accuracy += torch.sum(torch.eq(y_, y))
        batch_num += 1
        if batch_num >= 3:
            break
    print(accuracy / batch_size / batch_num)
