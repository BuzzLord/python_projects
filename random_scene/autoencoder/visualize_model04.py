from __future__ import print_function
import numpy as np
import cv2
from re import match
from os.path import exists, join

from model04 import *


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
        data_output, _ = model(data_input, position)
        loss = criterion(data_output, data_actual)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
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
    parser = argparse.ArgumentParser(description='PyTorch Model 04 Experiment')
    parser.add_argument('--file', type=str, metavar='FILENAME', required=True,
                        help='filename to visualize')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--load-model-state', type=str, default="", metavar='FILENAME',
                        help='filename to pre-trained model state to load')
    parser.add_argument('--model-path', type=str, default="model04", metavar='PATH',
                        help='pathname for this models output (default model04)')
    parser.add_argument('--log-file', type=str, default="", metavar='FILENAME',
                        help='filename to log output to')
    parser.add_argument('--dataset-resample', type=float, default=0.5, metavar='R',
                        help='resample/resize the dataset images (default 0.5)')
    parser.add_argument('--dataset-subsample', type=float, default=0.5, metavar='S',
                        help='subsample the dataset images to improve training (default 0.5)')
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

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    logging.info("Building dataset with resample rate {} and subsample rate {}".format(args.dataset_resample, args.dataset_subsample))
    dataset_transforms = transforms.Compose([dl.ResampleImages(args.dataset_resample),
                                             dl.SubsampleImages(args.dataset_subsample),
                                             dl.ToTensor(),
                                             dl.NormalizeImages(
                                                 mean=[0.5, 0.5, 0.5],
                                                 std=[0.5, 0.5, 0.5])
                                             ])

    model = Net(downsample_method='conv', upsample_method='conv', target_resolution=(512*args.dataset_resample, 512*args.dataset_resample))
    if len(args.load_model_state) > 0:
        model_path = os.path.join(args.model_path, args.load_model_state)
        if os.path.exists(model_path):
            logging.info("Loading model from {}".format(model_path))
            model.load_state_dict(torch.load(model_path))
    model = model.to(device)

    sample = load_sample(filename=args.file, root_dir='../screens_512', transform=dataset_transforms)
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
        save_image(gen_pic, './{}/image_{}.png'.format(args.model_path, os.path.splitext(args.file)[0]), nrow=4)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
