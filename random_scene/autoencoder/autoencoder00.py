from __future__ import print_function
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.models
import dataloader as dl


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # batch, 3, 256, 256

        # 256, 256, 3
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 128, 128, 32
        self.conv2 = nn.Conv2d(32, 48, kernel_size=3, padding=1)
        # 64, 64, 48
        self.iconv3 = nn.ConvTranspose2d(48, 24, kernel_size=3, padding=1)
        # 128, 128, 24
        self.iconv2 = nn.ConvTranspose2d(24, 8, kernel_size=3, padding=1)
        # 256, 256, 8
        self.iconv1 = nn.ConvTranspose2d(8, 3, kernel_size=3, padding=1)
        # 256, 256, 3

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(F.interpolate(self.iconv3(x), scale_factor=2.0))
        x = F.relu(F.interpolate(self.iconv2(x), scale_factor=2.0))
        x = self.iconv1(x)
        return torch.tanh(x)


def train(args, model, device, criterion, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data_actual = data['generated'].to(device)
        optimizer.zero_grad()
        data_output = model(data_actual)
        loss = criterion(data_output, data_actual)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    def to_img(x):
        x = x * 0.5
        x = x + 0.5
        x = x.clamp(0, 1)
        # x = x.view(x.size(0), x.size(1), x.size(2), x.size(3))
        return x

    if epoch % 1 == 0:
        images = torch.cat((data_output.cpu().data,data_actual.cpu().data), dim=2)
        gen_pic = to_img(images)
        save_image(gen_pic, './model00_img/autoencoder_image_{}.png'.format(epoch))


def test(args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data_actual = data['generated'].to(device)
            data_output = model(data_actual)
            test_loss += criterion(data_output, data_actual, reduction='sum').item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Model 00 AutoEncoder Experiment')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    dataset_transforms = transforms.Compose([dl.ResampleImages(0.5),
                                             # dl.SubsampleImages(0.25),
                                             dl.ToTensor(),
                                             dl.NormalizeImages(
                                                 mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
                                             ])
    train_loader = torch.utils.data.DataLoader(
        dl.RandomSceneDataset('../screens_256', transform=dataset_transforms),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dl.RandomSceneDataset('../test_256', transform=dataset_transforms),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, criterion, train_loader, optimizer, epoch)
        test(args, model, device, criterion, test_loader)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
