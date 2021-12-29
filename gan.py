import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.utils import save_image


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.actv = nn.LeakyReLU()
        self.logistic = nn.Sigmoid()

    def forward(self, x):
        y = self.actv(self.fc1(x))
        y = self.actv(self.fc2(y))
        y = self.actv(self.fc3(y))
        return self.logistic(y)


class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, output_size)
        self.actv = nn.LeakyReLU()
        self.norm = nn.Tanh()

    def forward(self, x):
        y = self.actv(self.fc1(x))
        y = self.actv(self.fc2(y))
        y = self.actv(self.fc3(y))
        return self.norm(self.fc4(y))


if __name__ == '__main__':
    run_in_colab = False
    if run_in_colab:
        dataset_path = '/content/drive/MyDrive/dataset'
        result_path = '/content/drive/MyDrive/result'
        discriminator_path = '/content/drive/MyDrive/model/discriminator.checkpoint'
        generator_path = '/content/drive/MyDrive/model/generator.checkpoint'
    else:
        dataset_path = './dataset'
        result_path = './result'
        discriminator_path = 'model/discriminator.checkpoint'
        generator_path = 'model/generator.checkpoint'
    batch_size = 100
    g_input_size = 128
    g_output_size = 28 * 28
    epoch_num = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = MNIST(dataset_path, train=True, transform=transforms.Compose([transforms.ToTensor(),
                                                                                  transforms.Normalize([0.5], [0.5])]))
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    d = Discriminator(g_output_size).to(device)
    g = Generator(g_input_size, g_output_size).to(device)
    loss_func = nn.BCELoss()
    d_optimizer = optim.Adam(d.parameters(), lr=1e-3)
    g_optimizer = optim.Adam(g.parameters(), lr=1e-3)
    label_true = torch.ones(batch_size, 1).to(device)
    label_false = torch.zeros(batch_size, 1).to(device)
    batch_num = len(train_loader)
    if os.path.isfile(discriminator_path) and os.path.isfile(generator_path):
        d.load_state_dict(torch.load(discriminator_path))
        g.load_state_dict(torch.load(generator_path))
    for i in range(epoch_num):
        avg_d_loss = 0
        avg_g_loss = 0
        for j, (x, _) in enumerate(train_loader):
            x = x.reshape(batch_size, -1).to(device)

            d_optimizer.zero_grad()
            y_true = d(x)
            loss_true = loss_func(y_true, label_true)
            rand_x = torch.randn(batch_size, g_input_size).to(device)
            y_false = d(g(rand_x))
            loss_false = loss_func(y_false, label_false)
            d_loss = loss_true + loss_false
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            rand_x = torch.randn(batch_size, g_input_size).to(device)
            g_result = g(rand_x)
            y_false = d(g_result)
            g_loss = loss_func(y_false, label_true)
            g_loss.backward()
            g_optimizer.step()

            avg_d_loss += d_loss.data
            avg_g_loss += g_loss.data

            if i % 10 == 9 and j + 1 == batch_num:
                result = g_result.reshape(batch_size, 1, 28, 28)
                save_image(result[:16], os.path.join(result_path, 'result_epoch_{}.png'.format(i)), nrow=4,
                           normalize=True)
        print('Epoch {}---- discriminator loss: {}; generator loss: {}'.format(i, avg_d_loss / batch_num,
                                                                               avg_g_loss / batch_num))
    torch.save(d.state_dict(), discriminator_path)
    torch.save(g.state_dict(), generator_path)
