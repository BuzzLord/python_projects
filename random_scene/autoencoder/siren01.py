from __future__ import print_function
import os
from collections import OrderedDict
from os.path import join
import argparse
import logging
import math
from statistics import mean, stdev
from random import shuffle

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
    #  Training regimes; all about the same number of parameters
    # = (10*L*H) + (O*H) + (N-2)*H*H
    # L:     10        10        10        10        10
    # H:    512       384       256       192       128
    # N:      4         6        10        16        34
    # O:      3         3         3         3         3
    #
    #    577024    629376    550656    535872    537472

    def __init__(self, hidden_size=256, hidden_layers=5, pos_encoding_levels=4):
        super(Siren, self).__init__()

        self.pos_encoding_levels = pos_encoding_levels
        # positional encoding (sin,cos) harmonics for five coords
        input_size = 2 * 5 * pos_encoding_levels
        output_size = 3

        self.w0_initial = 30.0
        self.hidden_layers = hidden_layers
        self.network = self.make_network(hidden_layers, hidden_size, input_size, output_size)

    @staticmethod
    def construct_layer(in_size, out_size, activation=None, c=6.0, w0=1.0):
        layers = OrderedDict([("linear", nn.Linear(in_size, out_size)),
                              ("activation", Sine(w0) if activation is None else activation)])
        std = math.sqrt(1 / in_size)
        c_std = math.sqrt(c) * std / w0
        layers["linear"].weight.data.uniform_(-c_std, c_std)
        layers["linear"].bias.data.uniform_(-std, std)
        return layers

    def make_network(self, hidden_layers, hidden_size, input_size, output_size):
        layers = [nn.Sequential(self.construct_layer(input_size, hidden_size, w0=self.w0_initial))]
        for i in range(2, hidden_layers):
            layers.append(nn.Sequential(self.construct_layer(hidden_size, hidden_size)))
        layers.append(nn.Sequential(self.construct_layer(hidden_size, output_size, activation=LinearActivation())))
        return nn.Sequential(OrderedDict([("layer{:d}".format(i + 1), layer) for i, layer in enumerate(layers)]))

    def forward(self, inputs):
        x = torch.cat([torch.cat((torch.sin(math.pow(2,i)*math.pi*inputs), torch.cos(math.pow(2,i)*math.pi*inputs)),
                                 dim=1) for i in range(self.pos_encoding_levels)], dim=1)
        out = self.network(x)
        return out

    def print_statistics(self):
        for k, m in self.named_modules():
            if isinstance(m, nn.Linear):
                logging.info("{}: weights {:.6f} ({:.6f})".format(k, m.weight.data.mean(), m.weight.data.std()))


def train(args, model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    train_set = train_loader.dataset
    for image_idx, image_filename in enumerate(train_loader):
        if args.print_statistics:
            model.print_statistics()
        data_loaders = train_set.generate_dataloaders(image_filename)

        batch_idx = 0
        total_batches = min([len(loader) for loader in data_loaders])
        loader_select = list(range(len(data_loaders)))

        loss_data = []
        while batch_idx < total_batches:
            shuffle(loader_select)
            samples = []
            for i in loader_select:
                try:
                    samples.append(next(data_loaders[i]))
                except StopIteration:
                    logging.error("Tried to load from empty data_loader {}".format(i))
            data_input = torch.cat([sample["inputs"] for sample in samples], 0).to(device, dtype=torch.float32)
            data_actual = torch.cat([sample["outputs"] for sample in samples], 0).to(device, dtype=torch.float32)

            optimizer.zero_grad()
            data_output = model(data_input)
            loss = criterion(data_output, data_actual)
            loss.backward()
            optimizer.step()
            loss_data.append(loss.item())
            batch_idx += 1

        if args.log_interval > 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3e} ({:.3e})'.format(
                epoch, image_idx+1, len(train_loader), 100. * (image_idx+1) / len(train_loader), mean(loss_data),
                stdev(loss_data)))
            if image_idx % 10 == 0:
                model.eval()
                torch.cuda.empty_cache()
                saver_loader = dl.RandomSceneSirenSampleSet(join(train_set.root_dir, image_filename[0]),
                                                            pos_scale=train_set.pos_scale, transform=train_set.transform)
                sample = saver_loader.get_in_order_sample()
                data_input = sample["inputs"]
                data_output = torch.zeros(sample["outputs"].shape)

                batch_indices = [(i, min(i+args.batch_size, data_output.shape[0]))
                                 for i in range(0, data_output.shape[0], args.batch_size)]

                for a, b in batch_indices:
                    data_input_row = data_input[a:b, :]
                    data_input_row = data_input_row.to(device, dtype=torch.float32)
                    data_output_row = model(data_input_row)
                    data_output[a:b, :] = data_output_row.cpu()

                data_actual = sample["outputs"].transpose(dim0=0, dim1=1).view((1, 3, 512, 512)).transpose(dim0=2, dim1=3)
                data_output = data_output.transpose(dim0=0, dim1=1).view((1, 3, 512, 512)).transpose(dim0=2, dim1=3).cpu()
                images = torch.cat((data_actual, data_output), dim=3).clamp(0, 1)
                save_image(images, join(args.model_path, "train{:02d}-{:02d}.png".format(epoch, int(image_idx/10))))
                model.train()


def test(args, model, device, test_loader, criterion, epoch):
    model.eval()
    with torch.no_grad():
        test_set = test_loader.dataset
        test_loss = []
        for image_idx, image_filename in enumerate(test_loader):
            data_loaders = test_set.generate_dataloaders(image_filename)

            batch_idx = 0
            total_batches = min([len(loader) for loader in data_loaders])
            loader_select = list(range(len(data_loaders)))

            while batch_idx < total_batches:
                samples = []
                for i in loader_select:
                    try:
                        samples.append(next(data_loaders[i]))
                    except StopIteration:
                        logging.error("Tried to load from empty data_loader {}".format(i))
                data_input = torch.cat([sample["inputs"] for sample in samples], 0).to(device, dtype=torch.float32)
                data_actual = torch.cat([sample["outputs"] for sample in samples], 0).to(device, dtype=torch.float32)

                data_output = model(data_input)
                test_loss.append(criterion(data_output, data_actual).item())
                batch_idx += 1

            if len(test_loss) > 1:
                loss_value = "{:.3e} ({:.3e})".format(mean(test_loss), stdev(test_loss))
            else:
                loss_value = "{:.3e}".format(test_loss[0])
            logging.info('Test set ({:.0f}%) loss: {}'.format(100. * (image_idx+1) / len(test_loader), loss_value))

            if image_idx % int(len(test_loader)/6) == 0:
                saver_loader = dl.RandomSceneSirenSampleSet(join(test_set.root_dir, image_filename[0]),
                                                            pos_scale=test_set.pos_scale, transform=test_set.transform)
                sample = saver_loader.get_in_order_sample()
                data_input = sample["inputs"].to(device, dtype=torch.float32)
                data_actual = sample["outputs"].to(device, dtype=torch.float32)
                data_output = model(data_input)

                data_actual = data_actual.transpose(dim0=0, dim1=1).view((1, 3, 512, 512)).transpose(dim0=2, dim1=3).cpu()
                data_output = data_output.transpose(dim0=0, dim1=1).view((1, 3, 512, 512)).transpose(dim0=2, dim1=3).cpu()
                images = torch.cat((data_actual, data_output), dim=3).clamp(0, 1)
                save_image(images, join(args.model_path, "test{:02d}-{:02d}.png".format(epoch, int(image_idx/int(len(test_loader)/6)))))

        if len(test_loss) > 1:
            loss_value = "{:.3e} ({:.3e})".format(mean(test_loss), stdev(test_loss))
        else:
            loss_value = "{:.3e}".format(test_loss[0])
        logging.info('Test set final loss: {}'.format(loss_value))


def main(custom_args=None):
    # Training settings
    model_number = "01"
    parser = argparse.ArgumentParser(description='PyTorch SIREN Model ' + model_number + ' Experiment')
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--test-batch-size', type=int, default=512, metavar='N',
                        help='input batch size for testing (default: 512)')
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
    parser.add_argument('--weight-decay', type=float, default=6e-7, metavar='D',
                        help='Weight decay (default: 6e-7)')
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
    parser.add_argument('--dataset', type=str, metavar='PATH',
                        help='training dataset path')
    parser.add_argument('--dataset-seed', type=str, metavar='SEED',
                        help='dataset seed number')
    parser.add_argument('--dataset-test-percent', type=float, metavar='PERCENT',
                        help='testing/validation dataset percent')

    parser.add_argument('--hidden-size', type=int, default=256, metavar='H',
                        help='Hidden layer size of Siren (default: 256)')
    parser.add_argument('--hidden-layers', type=int, default=5, metavar='N',
                        help='Hidden layer count of Siren (default: 5)')
    parser.add_argument('--pos-encoding', type=int, default=4, metavar='L',
                        help='Positional encoding harmonics (default: 4)')
    parser.add_argument('--print-statistics', action='store_true', default=False,
                        help='print out layer weight statistics periodically')
    args = parser.parse_args(args=custom_args)

    if args.model_path is None:
        args.model_path = join("siren{}".format(model_number),
                               "model_{:d}_{:d}_{:d}".format(args.pos_encoding, args.hidden_size, args.hidden_layers))

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    if len(args.log_file) > 0:
        log_formatter = logging.Formatter("%(asctime)s: %(message)s")
        root_logger = logging.getLogger()

        file_handler = logging.FileHandler(join(args.model_path, args.log_file))
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)

    logging.info("\n*** Starting Siren Model {}".format(model_number))
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    logging.info("Using random seed " + str(args.seed))
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    test_loader, train_loader, = get_data_loaders(args, kwargs)

    logging.info("Siren configured with pos_encoding = {}, hidden_size = {}, hidden_layers = {}".format(
        args.pos_encoding, args.hidden_size, args.hidden_layers))
    model = Siren(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers, pos_encoding_levels=args.pos_encoding)
    if len(args.load_model_state) > 0:
        model_path = os.path.join(args.model_path, args.load_model_state)
        if os.path.exists(model_path):
            logging.info("Loading model from {}".format(model_path))
            model.load_state_dict(torch.load(model_path))
            model.eval()
    model = model.to(device, dtype=torch.float32)

    logging.info("Using Adam optimizer with LR = {}, Beta = ({}, {}), Decay {}".format(args.lr, args.beta1, args.beta2,
                                                                                       args.weight_decay))
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, betas=(args.beta1, args.beta2),
                           weight_decay=args.weight_decay)
    criterion = ModelLoss(device=device)

    for epoch in range(args.epoch_start, args.epochs + args.epoch_start):
        train(args, model, device, train_loader, criterion, optimizer, epoch)
        test(args, model, device, test_loader, criterion, epoch)
        torch.save(model.state_dict(), join(args.model_path, "model_state_{}.pth".format(epoch)))


def get_data_loaders(args, kwargs):
    position_scale = [1 / 4, 1 / 3, 1 / 3]
    dataset_path = join('..', args.dataset)
    train_set = dl.RandomSceneSirenFileList(root_dir=dataset_path, dataset_seed=args.dataset_seed, is_test=False,
                                            batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True,
                                            pos_scale=position_scale)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)

    test_set = dl.RandomSceneSirenFileList(root_dir=dataset_path, dataset_seed=args.dataset_seed, is_test=True,
                                           batch_size=args.test_batch_size, num_workers=4, pin_memory=False,
                                           shuffle=False, pos_scale=position_scale)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

    return test_loader, train_loader


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(threadName)s] %(message)s')
    main()
