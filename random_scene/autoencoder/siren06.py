from __future__ import print_function
import os
import sys
from collections import OrderedDict
from os.path import join
import argparse
import multiprocessing
import logging
import math
from statistics import mean, stdev

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision.utils import save_image
import dataloader06 as dl


class ModelLoss(nn.Module):
    def __init__(self):
        super(ModelLoss, self).__init__()

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

    def expand_pos_encoding(self, inputs):
        pmult = math.pi * torch.pow(2, torch.linspace(-1, self.pos_encoding_levels[0], self.pos_encoding_levels[0],
                                                      device=inputs.device))
        pos = inputs[:, 0:3]
        u = torch.bmm(pmult.unsqueeze(1).repeat(inputs.shape[0], 1, 1), pos.unsqueeze(1))
        u = u.view((inputs.shape[0], 3 * self.pos_encoding_levels[0]))
        sin_u = torch.sin(u)
        cos_u = torch.cos(u)

        rmult = math.pi * torch.pow(2, torch.linspace(-1, self.pos_encoding_levels[1], self.pos_encoding_levels[1],
                                                      device=inputs.device))
        rot = inputs[:, 3:5]
        v = torch.bmm(rmult.unsqueeze(1).repeat(inputs.shape[0], 1, 1), rot.unsqueeze(1))
        v = v.view((inputs.shape[0],2*self.pos_encoding_levels[1]))
        sin_v = torch.sin(v)
        cos_v = torch.cos(v)
        x = torch.cat((sin_u, cos_u, sin_v, cos_v), dim=1)
        return x

    def forward(self, inputs):
        x = self.expand_pos_encoding(inputs)
        out = self.network(x)
        return out

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def print_statistics(self, logger):
        for k, m in self.named_modules():
            if isinstance(m, nn.Linear):
                logger.info("{}: weights {:.6f} ({:.6f})".format(k, m.weight.data.mean(), m.weight.data.std()))


def train(args, logger, model, rank, train_loader, criterion, optimizer, scalar, epoch):
    model.train()
    train_set = train_loader.dataset
    render_vector_count = 512*512
    for image_idx, sample_list in enumerate(train_loader):
        if args.print_statistics:
            model.print_statistics(logger)
        data_loader = train_set.generate_dataloader(sample_list, apply_transform=args.random_position,
                                                    max_t=(args.random_max_t*min((epoch-1)/args.random_steps, 1.0)))
        loss_data = []
        for batch_idx, sample in enumerate(data_loader):
            data_input = sample["inputs"]
            data_actual = sample["outputs"]

            model.zero_grad()
            with torch.cuda.amp.autocast():
                data_output = model(data_input)
                loss = criterion(data_output, data_actual)
            scalar.scale(loss).backward()
            scalar.step(optimizer)
            scalar.update()
            loss_data.append(loss.item())

        if args.log_interval > 0 and image_idx % args.log_interval == 0:
            if args.train_image_interval > 0 and (image_idx / args.log_interval) % args.train_image_interval == 0:
                with torch.no_grad():
                    torch.cuda.empty_cache()
                    sample = data_loader.get_in_order_sample()
                    data_input = sample["inputs"]
                    data_output = torch.zeros((0, 3), dtype=torch.float32)
                    for i in range(0, data_input.shape[0], render_vector_count):
                        j = min(data_input.shape[0], i + render_vector_count)
                        partial_output = model(data_input[i:j, :])
                        data_output = torch.cat((data_output, partial_output.cpu()), dim=0)

                    data_actual = convert_image(sample["outputs"].cpu(), sample["dims"])
                    data_output = convert_image(data_output, sample["dims"])
                    images = torch.cat((data_actual, data_output), dim=3)
                    save_image(images,
                               join(args.model_path,
                                    "train{:02d}-{:02d}.png".format(epoch, int(image_idx/10))),
                               nrow=1)
                    torch.cuda.empty_cache()

            m, s, b = get_loss_stats(loss_data)
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.3e} ({:.3e}, {:.3e})'.format(
                epoch, image_idx + 1, len(train_loader), 100. * (image_idx + 1) / len(train_loader), m, s, b))


def test(args, logger, model, rank, test_loader, criterion, epoch):
    model.eval()
    render_vector_count = 512 * 512
    img_save_modulus = max(1, int(len(test_loader)/6))
    with torch.no_grad():
        test_set = test_loader.dataset
        test_loss = []
        for image_idx, image_filename in enumerate(test_loader):
            data_loader = test_set.generate_dataloader(image_filename)

            for batch_idx, sample in enumerate(data_loader):
                data_input = sample["inputs"]
                data_actual = sample["outputs"]

                data_output = model(data_input)
                test_loss.append(criterion(data_output, data_actual).item())

            if image_idx % img_save_modulus == 0:
                torch.cuda.empty_cache()
                sample = data_loader.get_in_order_sample()
                data_input = sample["inputs"]
                data_output = torch.zeros((0, 3), dtype=torch.float32)
                for i in range(0, data_input.shape[0], render_vector_count):
                    j = min(data_input.shape[0], i + render_vector_count)
                    partial_output = model(data_input[i:j, :])
                    data_output = torch.cat((data_output, partial_output.cpu()), dim=0)

                sample_actual = convert_image(sample["outputs"].cpu(), sample["dims"])
                sample_output = convert_image(data_output, sample["dims"])
                images = torch.cat((sample_actual, sample_output), dim=3)
                save_image(images,
                           join(args.model_path,
                                "test{:02d}-{:02d}.png".format(epoch, int(image_idx/img_save_modulus))),
                           nrow=1)
            torch.cuda.empty_cache()

            if len(test_loss) > 1:
                m, s, b = get_loss_stats(test_loss)
                loss_value = "{:.3e} ({:.3e}, {:.3e})".format(m, s, b)
            else:
                loss_value = "{:.3e}".format(test_loss[0])
            logger.info('Test set ({:.0f}%) loss: {}'.format(100. * (image_idx+1) / len(test_loader), loss_value))


def convert_image(data, dims):
    if dims is None:
        dims = (256, round(data.shape[0] / 256))
    converted = data.transpose(dim0=0, dim1=1).view((1, 3, dims[0], dims[1]))
    converted = ((converted * 0.5) + 0.5).clamp(0, 1)
    return converted


def get_loss_stats(loss):
    """ Returns mean, standard deviation, and sample skewness. """
    n = len(loss)
    if n == 0:
        return 0, 0, 0
    elif n == 1:
        return loss[0], 0, 0
    else:
        c = mean(loss)
        s = stdev(loss, xbar=c)
        b = (sum((x - c) ** 3 for x in loss) / n) / pow(s, 3)
        return c, s, b


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_logger(args):
    logger = multiprocessing.get_logger()
    if not len(logger.handlers):
        logger.setLevel(logging.INFO)
        log_formatter = logging.Formatter("%(asctime)s: %(message)s")
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        logger.addHandler(stream_handler)

        if len(args.log_file) > 0:
            file_handler = logging.FileHandler(join(args.model_path, args.log_file))
            file_handler.setFormatter(log_formatter)
            logger.addHandler(file_handler)

    return logger


def get_data_loaders(args):
    position_scale = [1 / 4, 1 / 3, 1 / 3]
    dataset_path = [join('..', d) for d in args.dataset]
    use_dist = args.ddp_world_size > 1
    train_set = dl.RandomSceneSirenFileListLoader(root_dir=dataset_path, dataset_seed=args.dataset_seed, is_test=False,
                                                  batch_size=args.batch_size, num_workers=args.num_workers,
                                                  pin_memory=(not args.dont_pin_memory), shuffle=True,
                                                  pos_scale=position_scale, importance=args.importance)
    if use_dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.img_batch_size, sampler=train_sampler,
                                               shuffle=(train_sampler is None), collate_fn=train_set.collate_fn)

    test_batch_size = args.test_batch_size if args.test_batch_size is not None else args.batch_size
    test_set = dl.RandomSceneSirenFileListLoader(root_dir=dataset_path, dataset_seed=args.dataset_seed, is_test=True,
                                                 batch_size=test_batch_size, num_workers=args.num_workers,
                                                 pin_memory=(not args.dont_pin_memory), shuffle=False,
                                                 pos_scale=position_scale)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.img_batch_size, shuffle=False,
                                              collate_fn=test_set.collate_fn)

    return test_loader, train_loader, train_sampler


def arg_parser(input_args, model_number="06"):
    parser = argparse.ArgumentParser(description='PyTorch SIREN Model ' + model_number + ' Experiment')
    parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 2048)')
    parser.add_argument('--test-batch-size', type=int, metavar='N',
                        help='input batch size for testing (default: batch-size)')
    parser.add_argument('--img-batch-size', type=int, default=8, metavar='I',
                        help='number of images to pass to sample dataloader (default: 8)')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='number of workers to pass to the image dataloader (default: 4)')
    parser.add_argument('--dont-pin-memory', action='store_true', default=False,
                        help='do not pass pin_memory=True to image dataloader')
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
    parser.add_argument('--schedule-step', nargs='*', type=int, metavar='S',
                        help='Schedule steps for Multi-Step LR decay')
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
    parser.add_argument('--importance', type=float, metavar='IMP',
                        help='Importance sampling scalar')

    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--load-model-state', type=str, default="", metavar='FILENAME',
                        help='filename to pre-trained model state to load')
    parser.add_argument('--load-optim-state', type=str, default="", metavar='FILENAME',
                        help='filename to optimizer state to continue with')
    parser.add_argument('--load-scalar-state', type=str, default="", metavar='FILENAME',
                        help='filename to AMP scalar state to continue with')
    parser.add_argument('--model-path', type=str, metavar='PATH',
                        help='pathname for this models output (default siren' + model_number + ')')
    parser.add_argument('--log-interval', type=int, default=2, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--train-image-interval', type=int, default=0, metavar='N',
                        help='generate output images for every N log intervals')

    parser.add_argument('--log-file', type=str, default="", metavar='FILENAME',
                        help='filename to log output to')
    parser.add_argument('--dataset', nargs='+', type=str, metavar='PATH',
                        help='training dataset path')
    parser.add_argument('--dataset-seed', type=str, metavar='SEED',
                        help='dataset seed number')
    parser.add_argument('--dataset-test-percent', type=float, metavar='PERCENT',
                        help='testing/validation dataset percent')

    parser.add_argument('--ddp-world-size', type=int, default=1, metavar='NGPU',
                        help='number of GPUs to use in parallel')

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
    args = parser.parse_args(args=input_args)

    if args.model_path is None:
        args.model_path = join("siren{}".format(model_number),
                               "model_{:d}_{:d}_{:d}_{:d}".format(args.pos_encoding, args.rot_encoding,
                                                                  args.hidden_size, args.hidden_layers))
    return args


def run(rank, args):
    logger = get_logger(args)
    logger.info("Initializing rank {} process".format(rank))

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)

    use_dist = args.ddp_world_size > 1

    if use_dist:
        dist.init_process_group(backend="nccl", rank=rank, world_size=args.ddp_world_size)

    test_loader, train_loader, train_sampler = get_data_loaders(args)

    logger.info("Siren configured with pos_encoding = ({},{}), hidden_size = {}, hidden_layers = {}".format(
        args.pos_encoding, args.rot_encoding, args.hidden_size, args.hidden_layers))
    model = Siren(hidden_size=args.hidden_size, hidden_layers=args.hidden_layers,
                  pos_encoding_levels=(args.pos_encoding, args.rot_encoding), dropout=args.dropout)\
        .to(rank, dtype=torch.float32)
    if rank == 0:
        if len(args.load_model_state) > 0:
            model_path = os.path.join(args.model_path, args.load_model_state)
            if not os.path.exists(model_path):
                raise FileNotFoundError("Could not find model path {}".format(model_path))
            logger.info("Loading model from {}".format(model_path))
            model.load_state_dict(torch.load(model_path))
            model.eval()

    if use_dist:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    if len(args.load_optim_state) > 0:
        optim_path = os.path.join(args.model_path, args.load_optim_state)
        if not os.path.exists(optim_path):
            raise FileNotFoundError("Could not find optimizer path {}".format(optim_path))
        logger.info("Loading optimizer state from {}".format(optim_path))
        optimizer = optim.Adam(model.parameters())
        optimizer.load_state_dict(torch.load(optim_path))
    else:
        logger.info("Using Adam optimizer with LR = {}, Beta = ({}, {}), ".format(args.lr, args.beta1, args.beta2) +
                    "and Weight Decay {}".format(args.weight_decay))
        optimizer = optim.Adam(model.parameters(),
                               lr=args.lr, betas=(args.beta1, args.beta2),
                               weight_decay=args.weight_decay)

    if args.schedule_step is not None and len(args.schedule_step) > 0:
        logger.info("Using StepLR Learning Rate Scheduler " +
                    "with steps = {} and gamma = {}".format(args.schedule_step, args.schedule_gamma))
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule_step,
                                                   gamma=args.schedule_gamma, last_epoch=args.epoch_start-2)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.epochs], gamma=1.0)

    scalar = torch.cuda.amp.GradScaler()
    if len(args.load_scalar_state) > 0:
        scalar_path = os.path.join(args.model_path, args.load_scalar_state)
        if not os.path.exists(scalar_path):
            raise FileNotFoundError("Could not find scalar path {}".format(scalar_path))
        logger.info("Loading scalar state from {}".format(scalar_path))
        scalar.load_state_dict(torch.load(scalar_path))

    criterion = ModelLoss()

    for epoch in range(args.epoch_start, args.epochs + args.epoch_start):
        logger.info("Starting epoch {} with LR {:.3e}".format(epoch, get_lr(optimizer)))
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        train(args, logger, model, rank, train_loader, criterion, optimizer, scalar, epoch)
        test(args, logger, model, rank, test_loader, criterion, epoch)
        if rank == 0:
            torch.save(model.state_dict(), join(args.model_path, "model_state_{}.pth".format(epoch)))
            torch.save(optimizer.state_dict(), join(args.model_path, "optim_state_{}.pth".format(epoch)))
            torch.save(scalar.state_dict(), join(args.model_path, "scalar_state_{}.pth".format(epoch)))
        scheduler.step()

    if use_dist:
        dist.destroy_process_group()


def main(custom_args=None):
    # Training settings
    model_number = "06"
    args = arg_parser(custom_args, model_number)

    if not torch.cuda.is_available():
        raise RuntimeError("Cuda not available")

    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)

    logger = get_logger(args)

    logger.info("\n*** Starting Siren Model {}".format(model_number))
    logger.info("Arguments: {}".format(args))

    if args.ddp_world_size < 1:
        raise AssertionError("DDP world size < 1")
    elif args.ddp_world_size > torch.cuda.device_count():
        logger.error("DDP world size '{}' larger than GPU count '{}'!".format(
                     args.ddp_world_size, torch.cuda.device_count()))
        args.ddp_world_size = torch.cuda.device_count()

    logger.info("Using random seed " + str(args.seed))
    torch.manual_seed(args.seed)

    if dist.is_available():
        mp.spawn(run, args=(args,), nprocs=args.ddp_world_size, join=True)
    else:
        args.ddp_world_size = 1
        run(0, args)


if __name__ == '__main__':
    main()

