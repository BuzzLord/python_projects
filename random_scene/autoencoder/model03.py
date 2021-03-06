from __future__ import print_function
import os
import argparse
import logging
import math
from statistics import mean, stdev
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
import dataloader as dl


class SelfAttention(nn.Module):
    def __init__(self, inplanes, query_planes=None, resample_kernel=1):
        super().__init__()

        self.query_planes = query_planes or inplanes // 8

        self.query = nn.Conv1d(inplanes, self.query_planes, 1)
        self.key = nn.Conv1d(inplanes, self.query_planes, 1)
        self.value = nn.Conv1d(inplanes, inplanes, 1)

        self.gamma = nn.Parameter(torch.tensor(0.0))

        if resample_kernel > 1:
            self.downsample = nn.MaxPool2d(kernel_size=resample_kernel)
            self.upsample = UpsampleInterpolate(scale_factor=float(resample_kernel))
        else:
            self.downsample = None
            self.upsample = None

    def forward(self, input):
        x = input
        if self.downsample is not None:
            x = self.downsample(x)

        shape = x.shape
        flatten = x.view(shape[0], shape[1], -1)
        query = self.query(flatten).permute(0, 2, 1)
        key = self.key(flatten)
        value = self.value(flatten)
        query_key = torch.bmm(query, key)
        attn = F.softmax(query_key, 1)
        attn = torch.bmm(value, attn)
        attn = attn.view(*shape)
        if self.upsample is not None:
            attn = self.upsample(attn)
        out = self.gamma * attn + input

        return out


class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, downsample_method=None, attention_kernel=None):
        super(ConvBlock, self).__init__()

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
        self.activation = nn.ReLU(inplace=True)
        if attention_kernel is not None:
            self.attention = SelfAttention(planes, resample_kernel=attention_kernel)
        else:
            self.attention = None

    def forward(self, x):
        residual = x

        out = self.stage1(x)
        out = self.stage2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.activation(out)

        if self.attention is not None:
            out = self.attention(out)

        return out


class UpsampleInterpolate(nn.Module):
    def __init__(self, scale_factor=2.0):
        super(UpsampleInterpolate, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor)


class ConvTransposeBlock(nn.Module):

    def __init__(self, inplanes, planes, upsample_method=None, attention_kernel=None):
        super(ConvTransposeBlock, self).__init__()

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
        self.activation = nn.ReLU(inplace=True)
        if attention_kernel is not None:
            self.attention = SelfAttention(planes, resample_kernel=attention_kernel)
        else:
            self.attention = None

    def forward(self, x):
        out = self.stage1(x)
        out = self.stage2(out)
        out = self.activation(out)

        if self.attention is not None:
            out = self.attention(out)

        return out


class Net(nn.Module):
    def __init__(self, downsample_method='conv', upsample_method='conv'):
        super(Net, self).__init__()
        # width = 256, height = 256
        # 12 = RGB Left, RGB Right, XYZ* Position

        # downsample_method = ['maxpool', 'conv']
        # upsample_method = ['interpolate', 'conv']

        # Position upsample stage
        # Input: b, 3, 1, 1
        self.position_upsample = nn.Sequential(
            nn.ConvTranspose2d(3, 16, 2, stride=1, padding=0),                  # b, 16, 2, 2
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            ConvTransposeBlock(16, 16, upsample_method=upsample_method),   # b, 16, 4, 4
            ConvTransposeBlock(16, 24, upsample_method=upsample_method),   # b, 24, 8, 8
            ConvTransposeBlock(24, 32, upsample_method=upsample_method),   # b, 32, 16, 16
            ConvTransposeBlock(32, 24, upsample_method=upsample_method),   # b, 24, 32, 32
            ConvTransposeBlock(24, 16, upsample_method=upsample_method),   # b, 16, 64, 64
            ConvTransposeBlock(16, 16, upsample_method=upsample_method),   # b, 16, 128, 128
            nn.ConvTranspose2d(16, 6, kernel_size=3, stride=1, padding=1, bias=False)
        )  # b, 6, 128, 128

        # Input: b, 12, 256, 256
        self.encoder_block1 = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ConvBlock(32, 32, attention_kernel=4)
        )  # b, 32, 256, 256
        self.encoder_block2 = nn.Sequential(
            ConvBlock(32, 43, downsample_method=downsample_method),
            ConvBlock(43, 43, attention_kernel=2)
        )  # b, 43, 128, 128
        self.encoder_block3 = nn.Sequential(
            ConvBlock(43, 57, downsample_method=downsample_method),
            ConvBlock(57, 57, attention_kernel=1)
        )  # b, 57, 64, 64
        self.encoder_block4 = nn.Sequential(
            ConvBlock(57, 76, downsample_method=downsample_method),
            ConvBlock(76, 76)
        )  # b, 76, 32, 32
        self.encoder_block5 = nn.Sequential(
            ConvBlock(76, 101, downsample_method=downsample_method),
            ConvBlock(101, 101)
        )  # b, 101, 16, 16
        self.encoder_block6 = nn.Sequential(
            ConvBlock(101, 135, downsample_method=downsample_method),
            ConvBlock(135, 135)
        )  # b, 135, 8, 8

        # Input: b, 135, 8, 8
        self.decoder_block1 = nn.Sequential(
            ConvTransposeBlock(135, 101, upsample_method=upsample_method),
            ConvTransposeBlock(101, 101)
        )  # b, 101, 16, 16
        self.decoder_block2 = nn.Sequential(
            ConvTransposeBlock(101 * 2, 76, upsample_method=upsample_method),
            ConvTransposeBlock(76, 76)
        )  # b, 76, 32, 32
        self.decoder_block3 = nn.Sequential(
            ConvTransposeBlock(76 * 2, 57, upsample_method=upsample_method),
            ConvTransposeBlock(57, 57, attention_kernel=1)
        )  # b, 57, 64, 64
        self.decoder_block4 = nn.Sequential(
            ConvTransposeBlock(57 * 2, 43, upsample_method=upsample_method),
            ConvTransposeBlock(43, 43, attention_kernel=2)
        )  # b, 43, 128, 128
        self.decoder_block5 = nn.Sequential(
            ConvTransposeBlock(43 * 2, 32, upsample_method=upsample_method),
            ConvTransposeBlock(32, 32, attention_kernel=4)
        )  # b, 32, 256, 256
        self.decoder_block6 = nn.Sequential(
            ConvTransposeBlock(32 * 2, 32),
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

        return out, (px, x1, x2, x3, x4, x5, x6, z1, z2, z3, z4, z5)


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

    def forward(self, input, target, reduction='mean'):
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
        data_output, _ = model(data_input, position)
        loss = criterion(data_output, data_actual)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            # with torch.no_grad():
            #     ratios = [torch.mean((torch.abs(param.grad * args.lr) / (torch.abs(param) + 1e-9))).view(-1).item()
            #     for param in model.parameters()]
            #     logging.info("Min {:.4e}, max {:.4e}, mean: {:.4e}, stdev: {:.4e}".format(min(ratios),max(ratios),
            #     mean(ratios),stdev(ratios)))

    def to_img(x):
        return ((x * 0.5) + 0.5).clamp(0,1)

    if epoch % 1 == 0:
        novel_images = torch.cat((data_output.cpu().data,data_actual.cpu().data), dim=3)
        input_data = data_input.cpu().data
        eye_images = torch.cat((input_data[:,0:3,:,:],input_data[:,3:6,:,:]), dim=3)
        images = torch.cat((novel_images,eye_images), dim=2)
        gen_pic = to_img(images)
        save_image(gen_pic, './{}/image_{}.png'.format(args.model_path, epoch), nrow=4)


def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            data_input = torch.cat((data['left'], data['right']), dim=1).to(device)
            data_actual = data['generated'].to(device)
            position = data['position'].to(device)
            data_output, _ = model(data_input, position)
            test_loss += criterion(data_output, data_actual).item() * args.test_batch_size

    test_loss /= len(test_loader.dataset)
    logging.info('\nTest set: Average loss: {:.6f}\n'.format(test_loss))


def main(custom_args=None):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Model 03 Experiment')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--epoch-start', type=int, default=1, metavar='N',
                        help='epoch number to start counting from')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--beta1', type=float, default=0.9, metavar='B1',
                        help='Adam beta 1 (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='Adam beta 2 (default: 0.999)')
    parser.add_argument('--weight-decay', type=float, default=6e-7, metavar='D',
                        help='Weight decay (default: 6e-7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--use-sgd', action='store_true', default=False,
                        help='uses SGD instead of Adam')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--load-model-state', type=str, default="", metavar='FILENAME',
                        help='filename to pre-trained model state to load')
    parser.add_argument('--model-path', type=str, default="model03", metavar='PATH',
                        help='pathname for this models output (default model03)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-file', type=str, default="", metavar='FILENAME',
                        help='filename to log output to')
    args = parser.parse_args(args=custom_args)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if len(args.log_file) > 0:
        log_formatter = logging.Formatter("%(asctime)s: %(message)s")
        root_logger = logging.getLogger()

        file_handler = logging.FileHandler("{0}/{1}".format(args.model_path, args.log_file))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    logging.info("Using random seed " + str(args.seed))
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    dataset_resample = 0.5
    dataset_subsample = 1.0
    logging.info("Building dataset with resample rate {} and subsample rate {}".format(dataset_resample,dataset_subsample))
    dataset_transforms = transforms.Compose([dl.ResampleImages(dataset_resample),
                                             dl.SubsampleImages(dataset_subsample),
                                             dl.ToTensor(),
                                             dl.NormalizeImages(
                                                 mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
                                             ])
    train_set = dl.RandomSceneDataset('../screens_256', transform=dataset_transforms)
    test_set = dl.RandomSceneDataset('../test_256', transform=dataset_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net(downsample_method='conv', upsample_method='conv')
    if len(args.load_model_state) > 0:
        model_path = os.path.join(args.model_path, args.load_model_state)
        if os.path.exists(model_path):
            logging.info("Loading model from {}".format(model_path))
            model.load_state_dict(torch.load(model_path))
            model.eval()
    model = model.to(device)

    if args.use_sgd:
        logging.info("Using SGD optimizer with LR = {}, M = {}".format(args.lr, args.momentum))
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        logging.info("Using Adam optimizer with LR = {}, Beta = ({}, {}), Decay {}".format(args.lr, args.beta1, args.beta2, args.weight_decay))
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    criterion = ModelLoss(device=device, value_weight=0.7, edge_weight=0.3)
    logging.info("Model loss using value weight {} and edge weight {}".format(criterion.value_weight, criterion.edge_weight))

    for epoch in range(args.epoch_start, args.epochs + args.epoch_start):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test(args, model, device, test_loader, criterion)
        torch.save(model.state_dict(), "./{}/model_state_{}.pth".format(args.model_path, epoch))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
