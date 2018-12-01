from __future__ import print_function
import argparse
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import dataloader as dl


class BasicConvBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample_method=None):
        super(BasicConvBlock, self).__init__()

        if downsample_method == "conv":
            self.stage1 = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=2, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(planes)
            )
        elif downsample_method == "maxpool":
            self.stage1 = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2)
            )
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(planes)
            )
        else:
            self.stage1 = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
            )
            self.downsample = None

        self.stage2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.stage1(x)
        out = self.stage2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class UpsampleInterpolate(nn.Module):
    def __init__(self, scale_factor=2.0):
        super(UpsampleInterpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)


class BasicConvTransposeBlock(nn.Module):

    def __init__(self, inplanes, planes, upsample_method=None):
        super(BasicConvTransposeBlock, self).__init__()

        if upsample_method == "conv":
            self.stage1 = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2, padding=0, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
            )
        elif upsample_method == "interpolate":
            self.stage1 = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True),
                UpsampleInterpolate(scale_factor=2.0)
            )
        else:
            self.stage1 = nn.Sequential(
                nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(planes),
                nn.ReLU(inplace=True)
            )

        self.stage2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.relu(out)

        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # width = 256, height = 256
        # 9 = RGB Left, RGB Right, XYZ Position

        # downsample_method = 'maxpool'
        # upsample_method = 'interpolate'
        downsample_method = 'conv'
        upsample_method = 'conv'

        # Position upsample stage
        # Input: b, 3, 1, 1
        self.position_upsample = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 2, stride=1, padding=0),                  # b, 16, 2, 2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            BasicConvTransposeBlock(16, 16, upsample_method=upsample_method),   # b, 16, 4, 4
            BasicConvTransposeBlock(16, 16, upsample_method=upsample_method),   # b, 16, 8, 8
            BasicConvTransposeBlock(16, 16, upsample_method=upsample_method),   # b, 16, 16, 16
            BasicConvTransposeBlock(16, 16, upsample_method=upsample_method),   # b, 16, 32, 32
            BasicConvTransposeBlock(16, 16, upsample_method=upsample_method),   # b, 16, 64, 64
            BasicConvTransposeBlock(16, 16, upsample_method=upsample_method),   # b, 16, 128, 128
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        )  # b, 3, 128, 128

        # Input: b, 9, 256, 256
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConvBlock(32, 32)
        )  # b, 32, 256, 256
        self.encoder_block2 = nn.Sequential(
            BasicConvBlock(32, 43, downsample_method=downsample_method),
            BasicConvBlock(43, 43)
        )  # b, 43, 128, 128
        self.encoder_block3 = nn.Sequential(
            BasicConvBlock(43, 57, downsample_method=downsample_method),
            BasicConvBlock(57, 57)
        )  # b, 57, 64, 64
        self.encoder_block4 = nn.Sequential(
            BasicConvBlock(57, 76, downsample_method=downsample_method),
            BasicConvBlock(76, 76)
        )  # b, 76, 32, 32
        self.encoder_block5 = nn.Sequential(
            BasicConvBlock(76, 101, downsample_method=downsample_method),
            BasicConvBlock(101, 101)
        )  # b, 101, 16, 16
        self.encoder_block6 = nn.Sequential(
            BasicConvBlock(101, 135, downsample_method=downsample_method),
            BasicConvBlock(135, 135)
        )  # b, 135, 8, 8

        # Input: b, 135, 8, 8
        self.decoder_block1 = nn.Sequential(
            BasicConvTransposeBlock(135, 101, upsample_method=upsample_method),
            BasicConvTransposeBlock(101, 101)
        )  # b, 101, 16, 16
        self.decoder_block2 = nn.Sequential(
            BasicConvTransposeBlock(101 * 2, 76, upsample_method=upsample_method),
            BasicConvTransposeBlock(76, 76)
        )  # b, 76, 32, 32
        self.decoder_block3 = nn.Sequential(
            BasicConvTransposeBlock(76 * 2, 57, upsample_method=upsample_method),
            BasicConvTransposeBlock(57, 57)
        )  # b, 57, 64, 64
        self.decoder_block4 = nn.Sequential(
            BasicConvTransposeBlock(57 * 2, 43, upsample_method=upsample_method),
            BasicConvTransposeBlock(43, 43)
        )  # b, 43, 128, 128
        self.decoder_block5 = nn.Sequential(
            BasicConvTransposeBlock(43 * 2, 32, upsample_method=upsample_method),
            BasicConvTransposeBlock(32, 32)
        )  # b, 32, 256, 256
        self.decoder_block6 = nn.Sequential(
            BasicConvTransposeBlock(32 * 2, 32),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )  # b, 3, 256, 256

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, p):
        px = self.position_upsample(p)
        x0 = torch.cat((x, px), dim=1)

        x1 = self.encoder_block1(x0)
        x2 = self.encoder_block2(x1)
        x3 = self.encoder_block3(x2)
        x4 = self.encoder_block4(x3)
        x5 = self.encoder_block5(x4)
        x6 = self.encoder_block6(x5)

        z1 = self.decoder_block1(x6)
        h2 = torch.cat((z1, x5), dim=1)
        z2 = self.decoder_block2(h2)
        h3 = torch.cat((z2, x4), dim=1)
        z3 = self.decoder_block3(h3)
        h4 = torch.cat((z3, x3), dim=1)
        z4 = self.decoder_block4(h4)
        h5 = torch.cat((z4, x2), dim=1)
        z5 = self.decoder_block5(h5)
        h6 = torch.cat((z5, x1), dim=1)
        out = self.decoder_block6(h6)

        return out


class ModelLoss(nn.Module):
    def __init__(self, device, value_weight=0.9, edge_weight=0.1):
        super(ModelLoss, self).__init__()
        edge_filter = ModelLoss.generate_filter()
        self.log_filter = edge_filter.to(device)
        self.value_weight = value_weight
        self.edge_weight = edge_weight

    @staticmethod
    def generate_filter():
        f = Variable(torch.FloatTensor([[[[-1 / 8, -1 / 8, -1 / 8],
                                          [-1 / 8,  8 / 8, -1 / 8],
                                          [-1 / 8, -1 / 8, -1 / 8]]]]),
                     requires_grad=False)
        return torch.cat((f, f, f), dim=1)

    def forward(self, input, target, reduction='elementwise_mean'):
        value_l1_loss = F.l1_loss(input, target, reduction=reduction)
        input_log_edges = F.conv2d(input, self.log_filter, padding=1)
        target_log_edges = F.conv2d(target, self.log_filter, padding=1)
        edge_l1_loss = F.l1_loss(input_log_edges, target_log_edges, reduction=reduction)
        return value_l1_loss * self.value_weight + edge_l1_loss * self.edge_weight


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data_input = torch.cat((data['left'], data['right']), dim=1).to(device)
        data_actual = data['generated'].to(device)
        position = data['position'].to(device)
        optimizer.zero_grad()
        data_output = model(data_input, position)
        loss = criterion(data_output, data_actual)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    def to_img(x):
        return ((x * 0.5) + 0.5).clamp(0,1)

    if epoch % 1 == 0:
        novel_images = torch.cat((data_output.cpu().data,data_actual.cpu().data), dim=3)
        input_data = data_input.cpu().data
        eye_images = torch.cat((input_data[:,0:3,:,:],input_data[:,3:6,:,:]), dim=3)
        images = torch.cat((novel_images,eye_images), dim=2)
        gen_pic = to_img(images)
        save_image(gen_pic, './model02_img/image_{}.png'.format(epoch), nrow=4)


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data_input = torch.cat((data['left'], data['right']), dim=1).to(device)
            data_actual = data['generated'].to(device)
            position = data['position'].to(device)
            data_output = model(data_input, position)
            test_loss += criterion(data_output, data_actual, reduction='sum').item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))


def main(custom_args=None):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Model 02 Experiment')
    parser.add_argument('--batch-size', type=int, default=24, metavar='N',
                        help='input batch size for training (default: 24)')
    parser.add_argument('--test-batch-size', type=int, default=48, metavar='N',
                        help='input batch size for testing (default: 48)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.9, metavar='B1',
                        help='Adam beta 1 (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='Adam beta 2 (default: 0.999)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--use-sgd', action='store_true', default=False,
                        help='uses SGD instead of Adam')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args(args=custom_args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    dataset_transforms = transforms.Compose([dl.ResampleImages(0.5),
                                             dl.SubsampleImages(0.25),
                                             dl.ToTensor(),
                                             dl.NormalizeImages(
                                                 mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
                                             ])
    train_set = dl.RandomSceneDataset('../screens_256', transform=dataset_transforms)
    test_set = dl.RandomSceneDataset('../test_256', transform=dataset_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    if args.use_sgd:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    criterion = ModelLoss(device=device)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test(args, model, device, test_loader, criterion)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
