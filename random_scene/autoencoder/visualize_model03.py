from __future__ import print_function
import numpy as np
import cv2
from re import match
from os.path import exists, join

from model03 import *


class NetLocal(nn.Module):
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

        return out


def load_sample(filename, root_dir=".", transform=None):
    fg = match("ss1_([0-9]*)([lrg])_([0-9.\-]*)_([0-9.\-]*)_([0-9.\-]*)\.png", filename)
    seed = fg.group(1)
    left_name = "ss1_" + seed + "l_-1.00_0.00_0.00.png"
    if not exists(join(root_dir, left_name)):
        logging.error("Missing left file: " + left_name)
        raise Exception("Missing left file: " + left_name)
    right_name = "ss1_" + seed + "r_1.00_0.00_0.00.png"
    if not exists(join(root_dir, right_name)):
        logging.error("Missing right file: " + right_name)
        raise Exception("Missing right file: " + right_name)
    position = np.array([float(fg.group(3)), float(fg.group(4)), float(fg.group(5))], dtype=np.float32)

    scale = (1.0 / 255.0)
    generated_image = np.clip(scale * np.array(cv2.imread(join(root_dir, filename)), dtype=np.float32), 0.0, 1.0)[:, :, ::-1]
    left_image = np.clip(scale * np.array(cv2.imread(join(root_dir, left_name)), dtype=np.float32), 0.0, 1.0)[:, :, ::-1]
    right_image = np.clip(scale * np.array(cv2.imread(join(root_dir, right_name)), dtype=np.float32), 0.0, 1.0)[:, :, ::-1]
    sample = {
        'generated_name': filename,
        'generated': generated_image,
        'left_name': left_name,
        'left': left_image,
        'right_name': right_name,
        'right': right_image,
        'position': position
    }

    if transform:
        sample = transform(sample)

    return sample


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
            data_output = model(data_input, position)
            test_loss += criterion(data_output, data_actual).item() * args.test_batch_size

    test_loss /= len(test_loader.dataset)
    logging.info('\nTest set: Average loss: {:.6f}\n'.format(test_loss))


def main(custom_args=None):
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Model 03 Experiment')

    parser.add_argument('--file', type=str, metavar='FILENAME', required=True,
                        help='filename to visualize')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--load-model-state', type=str, default="", metavar='FILENAME',
                        help='filename to pre-trained model state to load')
    parser.add_argument('--model-path', type=str, default="vis_model03", metavar='PATH',
                        help='pathname for this models output (default vis_model03)')
    parser.add_argument('--log-file', type=str, default="", metavar='FILENAME',
                        help='filename to log output to')
    args = parser.parse_args(args=custom_args)

    if len(args.log_file) > 0:
        log_formatter = logging.Formatter("%(asctime)s: %(message)s")
        root_logger = logging.getLogger()

        file_handler = logging.FileHandler("{0}/{1}".format(args.model_path, args.log_file))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    dataset_resample = 0.5
    dataset_subsample = 0.5
    logging.info("Building dataset with resample rate {} and subsample rate {}".format(dataset_resample,dataset_subsample))
    dataset_transforms = transforms.Compose([dl.ResampleImages(dataset_resample),
                                             dl.SubsampleImages(dataset_subsample),
                                             dl.ToTensor(),
                                             dl.NormalizeImages(
                                                 mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
                                             ])

    model = Net(downsample_method='conv', upsample_method='conv')
    if len(args.load_model_state) > 0:
        model_path = os.path.join(args.model_path, args.load_model_state)
        if os.path.exists(model_path):
            logging.info("Loading model from {}".format(model_path))
            model.load_state_dict(torch.load(model_path))
            model.eval()
    model = model.to(device)

    sample = load_sample(filename=args.file, root_dir='../screens_256', transform=dataset_transforms)
    model.eval()
    with torch.no_grad():
        data_input = torch.cat((sample['left'], sample['right']), dim=0).unsqueeze(0).to(device)
        data_actual = sample['generated'].unsqueeze(0).to(device)
        position = sample['position'].unsqueeze(0).to(device)
        data_output, intermediates = model(data_input, position)

        def to_img(x):
            return ((x * 0.5) + 0.5).clamp(0, 1)

        novel_images = torch.cat((data_output.cpu().data, data_actual.cpu().data), dim=3)
        input_data = data_input.cpu().data
        eye_images = torch.cat((input_data[:, 0:3, :, :], input_data[:, 3:6, :, :]), dim=3)
        images = torch.cat((novel_images, eye_images), dim=2)
        gen_pic = to_img(images)
        save_image(gen_pic, './{}/image_{}.png'.format(args.model_path, args.file), nrow=4)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
