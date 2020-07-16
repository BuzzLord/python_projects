from __future__ import print_function
import os
from collections import OrderedDict
from os.path import join
import argparse
import logging
import math
from statistics import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import dataloader as dl


class ModelLoss(nn.Module):
    def __init__(self, device):
        super(ModelLoss, self).__init__()
        self.device = device

    def forward(self, inputs, target):
        loss = F.mse_loss(inputs, target)
        return loss


class Sine(nn.Module):
    def __init__(self, w0=1.):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class LinearActivation(nn.Module):
    def __init__(self):
        super(LinearActivation, self).__init__()

    def forward(self, x):
        return x


class Siren(nn.Module):
    def __init__(self, hidden_size=256, hidden_layers=5, pos_encoding_levels=None, dropout=None):
        super(Siren, self).__init__()

        if pos_encoding_levels is None:
            self.pos_encoding_levels = (4, 4)
        self.pos_encoding_levels = pos_encoding_levels
        # positional encoding (sin,cos) harmonics for five coords; 0 is position, 1 is rotation
        input_size = 2 * 3 * pos_encoding_levels[0] + 2 * 2 * pos_encoding_levels[1]
        output_size = 3

        self.w0_initial = 30.0
        self.hidden_layers = hidden_layers
        self.network = self.make_network(hidden_layers, hidden_size, input_size, output_size, dropout)

    @staticmethod
    def construct_layer(in_size, out_size, activation=None, c=6.0, w0=1.0, dropout=None):
        layers = OrderedDict([("linear", nn.Linear(in_size, out_size))])
        if dropout is not None:
            layers["dropout"] = nn.Dropout(p=dropout)
        layers["activation"] = Sine(w0) if activation is None else activation

        std = math.sqrt(1 / in_size)
        c_std = math.sqrt(c) * std / w0
        layers["linear"].weight.data.uniform_(-c_std, c_std)
        layers["linear"].bias.data.uniform_(-std, std)
        return layers

    def make_network(self, hidden_layers, hidden_size, input_size, output_size, dropout):
        layers = [nn.Sequential(self.construct_layer(input_size, hidden_size, w0=self.w0_initial, dropout=dropout))]
        for i in range(2, hidden_layers):
            layers.append(nn.Sequential(self.construct_layer(hidden_size, hidden_size, dropout=dropout)))
        layers.append(nn.Sequential(self.construct_layer(hidden_size, output_size, activation=LinearActivation())))
        return nn.Sequential(OrderedDict([("layer{:d}".format(i + 1), layer) for i, layer in enumerate(layers)]))

    def forward(self, inputs):
        xp = torch.cat([torch.cat((torch.sin(math.pow(2,i)*math.pi*inputs[:, 0:3]),
                                  torch.cos(math.pow(2,i)*math.pi*inputs[:, 0:3])),
                                  dim=1) for i in range(self.pos_encoding_levels[0])], dim=1)
        xr = torch.cat([torch.cat((torch.sin(math.pow(2, i) * math.pi * inputs[:, 3:5]),
                                  torch.cos(math.pow(2, i) * math.pi * inputs[:, 3:5])),
                                  dim=1) for i in range(self.pos_encoding_levels[1])], dim=1)
        x = torch.cat((xp, xr), dim=1)
        out = self.network(x)
        return out

    def print_statistics(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Linear):
                logging.info("{}: weights {:.6f} ({:.6f})".format(k, m.weight.data.mean(), m.weight.data.std()))


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_set = train_loader.dataset
    transform = dl.SirenSampleRandomizePosition(args.random_position,
                                                max_t=(args.random_max_t*min((epoch-1)/args.random_steps, 1.0)))
    for image_idx, image_filename in enumerate(train_loader):
        if args.print_statistics:
            model.print_statistics()
        data_loader = train_set.generate_dataloader(image_filename)

        loss_data = []
        for batch_idx, sample in enumerate(data_loader):
            data_input = transform.vector_transform(sample["inputs"].to(device, dtype=torch.float32))
            data_actual = sample["outputs"].to(device, dtype=torch.float32)

            optimizer.zero_grad()
            data_output = model(data_input)
            loss = criterion(data_output, data_actual)
            loss.backward()
            optimizer.step()
            loss_data.append(loss.item())

        if args.log_interval > 0:
            if image_idx % 10 == 0:
                with torch.no_grad():
                    sample = data_loader.dataset.get_in_order_sample()
                    data_input = sample["inputs"].to(device, dtype=torch.float32)
                    data_output = model(data_input)

                    data_actual = convert_image(sample["outputs"], sample["dims"])
                    data_output = convert_image(data_output.cpu(), sample["dims"])
                    images = torch.cat((data_actual, data_output), dim=3)
                    save_image(images,
                               join(args.model_path,
                                    "train{:02d}-{:02d}.png".format(epoch, int(image_idx/10))),
                               nrow=1)
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3e} ({:.3e})'.format(
                epoch, image_idx + 1, len(train_loader), 100. * (image_idx + 1) / len(train_loader), mean(loss_data),
                stdev(loss_data)))


def test(args, model, device, test_loader, criterion, epoch):
    model.eval()
    with torch.no_grad():
        test_set = test_loader.dataset
        test_loss = []
        for image_idx, image_filename in enumerate(test_loader):
            data_loader = test_set.generate_dataloader(image_filename)

            for batch_idx, sample in enumerate(data_loader):
                data_input = sample["inputs"].to(device, dtype=torch.float32)
                data_actual = sample["outputs"].to(device, dtype=torch.float32)

                data_output = model(data_input)
                test_loss.append(criterion(data_output, data_actual).item())

            if image_idx % int(len(test_loader)/6) == 0:
                sample = data_loader.dataset.get_in_order_sample()
                data_input = sample["inputs"].to(device, dtype=torch.float32)
                data_output = model(data_input)

                sample_actual = convert_image(sample["outputs"], sample["dims"])
                sample_output = convert_image(data_output.cpu(), sample["dims"])
                images = torch.cat((sample_actual, sample_output), dim=3)
                save_image(images,
                           join(args.model_path,
                                "test{:02d}-{:02d}.png".format(epoch, int(image_idx/int(len(test_loader)/6)))),
                           nrow=1)

            if len(test_loss) > 1:
                loss_value = "{:.3e} ({:.3e})".format(mean(test_loss), stdev(test_loss))
            else:
                loss_value = "{:.3e}".format(test_loss[0])
            logging.info('Test set ({:.0f}%) loss: {}'.format(100. * (image_idx+1) / len(test_loader), loss_value))


def convert_image(data, dims):
    converted = data.transpose(dim0=0, dim1=1).view((1, 3, dims[0], dims[1]))
    converted = ((converted * 0.5) + 0.5).clamp(0, 1)
    return converted


def main(custom_args=None):
    # Training settings
    model_number = "02"
    parser = argparse.ArgumentParser(description='PyTorch SIREN Model ' + model_number + ' Experiment')
    parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--test-batch-size', type=int, metavar='N',
                        help='input batch size for testing (default: batch-size)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--epoch-start', type=int, default=1, metavar='N',
                        help='epoch number to start counting from')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.0001)')
    parser.add_argument('--beta1', type=float, default=0.9, metavar='B1',
                        help='Adam beta 1 (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, metavar='B2',
                        help='Adam beta 2 (default: 0.999)')
    parser.add_argument('--weight-decay', type=float, default=0., metavar='D',
                        help='Weight decay (default: 0.)')
    parser.add_argument('--schedule-step-size', type=int, default=4, metavar='S',
                        help='Schedule step size for LR decay (default: 4)')
    parser.add_argument('--schedule-gamma', type=float, default=1.0, metavar='G',
                        help='Schedule gamma factor for LR decay (default: 1.0)')
    parser.add_argument('--dropout', type=float, metavar='DROPOUT',
                        help='Dropout rate for each linear layer (except last)')

    parser.add_argument('--random-position', action='store_true', default=False,
                        help='Randomize position along view ray during training')
    parser.add_argument('--random-steps', type=int, default=5, metavar='S',
                        help='Randomize position number of steps to max_t (default: 5)')
    parser.add_argument('--random-max-t', type=float, default=0.5, metavar='T',
                        help='Randomize position max t (default: 0.5)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--load-model-state', type=str, default="", metavar='FILENAME',
                        help='filename to pre-trained model state to load')
    parser.add_argument('--model-path', type=str, metavar='PATH',
                        help='pathname for this models output (default siren'+model_number+')')
    parser.add_argument('--log-interval', type=int, default=128, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--log-file', type=str, default="", metavar='FILENAME',
                        help='filename to log output to')
    parser.add_argument('--dataset', nargs='+', type=str, metavar='PATH',
                        help='training dataset path')
    parser.add_argument('--dataset-seed', type=str, metavar='SEED',
                        help='dataset seed number')
    parser.add_argument('--dataset-test-percent', type=float, metavar='PERCENT',
                        help='testing/validation dataset percent')

    parser.add_argument('--hidden-size', type=int, default=256, metavar='H',
                        help='Hidden layer size of Siren (default: 256)')
    parser.add_argument('--hidden-layers', type=int, default=5, metavar='N',
                        help='Hidden layer count of Siren (default: 5)')
    parser.add_argument('--pos-encoding', type=int, default=6, metavar='LP',
                        help='Positional encoding harmonics (default: 6)')
    parser.add_argument('--rot-encoding', type=int, default=4, metavar='LR',
                        help='Rotational encoding harmonics (default: 4)')
    parser.add_argument('--print-statistics', action='store_true', default=False,
                        help='print out layer weight statistics periodically')
    args = parser.parse_args(args=custom_args)

    if args.model_path is None:
        args.model_path = join("siren{}".format(model_number),
                               "model_{:d}_{:d}_{:d}_{:d}".format(args.pos_encoding, args.rot_encoding,
                                                                  args.hidden_size, args.hidden_layers))

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if len(args.log_file) > 0:
        log_formatter = logging.Formatter("%(asctime)s: %(message)s")
        root_logger = logging.getLogger()

        file_handler = logging.FileHandler(join(args.model_path, args.log_file))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    logging.info("\n*** Starting Siren Model {}".format(model_number))
    logging.info("Arguments: {}".format(args))
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    logging.info("Using random seed " + str(args.seed))
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_loader, train_loader, = get_data_loaders(args, kwargs)

    logging.info("Siren configured with pos_encoding = ({},{}), hidden_size = {}, hidden_layers = {}".format(
        args.pos_encoding, args.rot_encoding, args.hidden_size, args.hidden_layers))
    model = Siren(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                  pos_encoding_levels=(args.pos_encoding, args.rot_encoding), dropout=args.dropout)
    if len(args.load_model_state) > 0:
        model_path = os.path.join(args.model_path, args.load_model_state)
        if os.path.exists(model_path):
            logging.info("Loading model from {}".format(model_path))
            model.load_state_dict(torch.load(model_path))
            model.eval()
    model = model.to(device, dtype=torch.float32)

    logging.info("Using Adam optimizer with LR = {}, Beta = ({}, {}), ".format(args.lr, args.beta1, args.beta2) +
                 "and Weight Decay {}".format(args.weight_decay))
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay)
    logging.info("Using StepLR Learning Rate Scheduler " +
                 "with step size = {} and gamma = {}".format(args.schedule_step_size, args.schedule_gamma))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.schedule_step_size, gamma=args.schedule_gamma)
    criterion = ModelLoss(device=device)

    for epoch in range(args.epoch_start, args.epochs + args.epoch_start):
        logging.info("Starting epoch {} with LR {:.3e}".format(epoch, scheduler.get_last_lr()[0]))
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test(args, model, device, test_loader, criterion, epoch)
        torch.save(model.state_dict(), join(args.model_path, "model_state_{}.pth".format(epoch)))
        scheduler.step()


def get_data_loaders(args, kwargs):
    position_scale = [1 / 4, 1 / 3, 1 / 3]
    dataset_path = [join('..', d) for d in args.dataset]
    train_set = dl.RandomSceneSirenFileList(root_dir=dataset_path, dataset_seed=args.dataset_seed, is_test=False,
                                            batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True,
                                            pos_scale=position_scale, importance=4.0)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)

    test_batch_size = args.test_batch_size if args.test_batch_size is not None else args.batch_size
    test_set = dl.RandomSceneSirenFileList(root_dir=dataset_path, dataset_seed=args.dataset_seed, is_test=True,
                                           batch_size=test_batch_size, num_workers=4, pin_memory=True,
                                           shuffle=False, pos_scale=position_scale)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

    return test_loader, train_loader


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(threadName)s] %(message)s')
    main()
