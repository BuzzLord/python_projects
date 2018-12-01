from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from dataloader import RandomSceneDataset, ToTensor, NormalizeImages


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # width = 512, height = 512
        # (3 Left, 3 Right, 3 Position)
        self.input_dim = 1

        # 28, 28, 1
        self.layer_1_size = 16
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        # 14, 14, 16
        self.layer_2_size = 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # self.bnormc2 = nn.BatchNorm2d(self.layer_2_size)
        # 7, 7, 32

        self.iconv3 = nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1)
        # self.bnormi2 = nn.BatchNorm2d(self.layer_1_size)
        self.iconv2 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=1)
        self.iconv1 = nn.ConvTranspose2d(16, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        # x = F.relu(self.bnormc2(F.max_pool2d(self.conv2(x), kernel_size=2)))
        # x = F.relu(self.bnormc3(F.max_pool2d(self.conv3(x), kernel_size=2)))
        # x = F.relu(self.bnormc4(self.conv4(x)))
        # x = F.relu(self.bnormi4(F.interpolate(self.iconv4(x), scale_factor=2.0)))
        # x = F.relu(self.bnormi3(F.interpolate(self.iconv3(x), scale_factor=2.0)))
        # x = F.relu(self.bnormi2(F.interpolate(self.iconv2(x), scale_factor=2.0)))
        x = F.relu(F.interpolate(self.iconv3(x), scale_factor=2.0))
        x = F.relu(F.interpolate(self.iconv2(x), scale_factor=2.0))
        x = self.iconv1(x)
        return torch.tanh(x)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, data)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    def to_img(x):
        x = x * 0.5
        x = x + 0.5
        x = x.clamp(0, 1)
        x = x.view(x.size(0), x.size(1), x.size(2), x.size(3))
        return x

    if epoch % 1 == 0:
        images = torch.cat((output.cpu().data,data.cpu().data), dim=2)
        gen_pic = to_img(images)
        save_image(gen_pic, './dc_img/image_{}.png'.format(epoch))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, data, reduction='sum').item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
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

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)



if __name__ == '__main__':
    main()