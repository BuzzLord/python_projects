from os import listdir
from os.path import isfile, join, exists
from re import match
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import cv2
import matplotlib.pyplot as plt
import logging


class ExpandPosition(object):
    """Expand the 3x1 position vector into a 3xHxW image/numpy array."""

    def __call__(self, sample):
        generated_image = sample['generated']
        position = sample['position']
        expanded_position = position * np.ones((generated_image.shape[0],generated_image.shape[1],position.size), dtype=np.float32)

        return {
            'generated_name': sample['generated_name'],
            'generated': sample['generated'],
            'left_name': sample['left_name'],
            'left': sample['left'],
            'right_name': sample['right_name'],
            'right': sample['right'],
            'position': sample['position'],
            'expanded_position': expanded_position
        }


class SubsampleImages(object):
    """Downsample then upsample the images in the sample by scale."""

    def __init__(self, scale, inter_down=cv2.INTER_AREA, inter_up=cv2.INTER_LINEAR):
        self.scale = scale
        self.inter_down = inter_down
        self.inter_up = inter_up

    def __call__(self, sample):
        def rescale(img):
            orig_dim = (img.shape[1], img.shape[0])
            scale_dim = (int(orig_dim[0] * self.scale), int(orig_dim[1] * self.scale))
            if scale_dim[0] == orig_dim[0] and scale_dim[1] == orig_dim[1]:
                return img
            return cv2.resize(
                cv2.resize(img, scale_dim, interpolation=self.inter_down),
                orig_dim, interpolation=self.inter_up)

        resampled = {
                'generated_name': sample['generated_name'],
                'generated': rescale(sample['generated']),
                'left_name': sample['left_name'],
                'left': rescale(sample['left']),
                'right_name': sample['right_name'],
                'right': rescale(sample['right']),
                'position': sample['position']
            }
        if 'expanded_position' in sample:
            resampled['expanded_position'] = sample['expanded_position']
        return resampled


class ResampleImages(object):
    """Downsample or upsample the images in the sample by scale."""

    def __init__(self, scale, interpolation=cv2.INTER_AREA):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, sample):
        def rescale(img):
            scale_dim = (int(img.shape[1] * self.scale), int(img.shape[0] * self.scale))
            return cv2.resize(img, scale_dim, interpolation=self.interpolation)

        resampled = {
                'generated_name': sample['generated_name'],
                'generated': rescale(sample['generated']),
                'left_name': sample['left_name'],
                'left': rescale(sample['left']),
                'right_name': sample['right_name'],
                'right': rescale(sample['right']),
                'position': sample['position']
            }
        if 'expanded_position' in sample:
            resampled['expanded_position'] = sample['expanded_position']
        return resampled


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print(str(dt.now()) + " ToTensoring " + sample['generated_name'])

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        resampled = {
            'generated_name': sample['generated_name'],
            'generated': torch.from_numpy(sample['generated'].transpose((2, 0, 1))),
            'left_name': sample['left_name'],
            'left': torch.from_numpy(sample['left'].transpose((2, 0, 1))),
            'right_name': sample['right_name'],
            'right': torch.from_numpy(sample['right'].transpose((2, 0, 1))),
            'position': torch.reshape(torch.from_numpy(sample['position']), (3, 1, 1))
        }
        if 'expanded_position' in sample:
            resampled['expanded_position'] = torch.from_numpy(sample['expanded_position'].transpose((2, 0, 1)))
        return resampled


class NormalizeImages(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Args:
            sample (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        # print(str(dt.now()) + " Normalizing " + sample['generated_name'])
        generated_image = sample['generated']
        left_image, right_image = sample['left'], sample['right']

        for t, m, s in zip(generated_image, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(left_image, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(right_image, self.mean, self.std):
            t.sub_(m).div_(s)
        # print(str(dt.now()) + " Finished " + sample['generated_name'])

        resampled = {
            'generated_name': sample['generated_name'],
            'generated': generated_image,
            'left_name': sample['left_name'],
            'left': left_image,
            'right_name': sample['right_name'],
            'right': right_image,
            'position': sample['position']
        }
        if 'expanded_position' in sample:
            resampled['expanded_position'] = sample['expanded_position']
        return resampled


class RandomSceneDataset(Dataset):
    """Random scene dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.scene_data = []
        logging.info("Checking dataset directory listing.")
        file_list = [f for f in listdir(self.root_dir) if isfile(join(self.root_dir, f)) and f.startswith("ss1") and f.endswith(".png")]
        logging.info("Found " + str(len(file_list)) + " files. Collating filenames into list.")
        for f in file_list:
            fg = match("ss1_([0-9]*)([lrg])_([0-9.\-]*)_([0-9.\-]*)_([0-9.\-]*)\.png", f)
            seed = fg.group(1)
            left_name = "ss1_" + seed + "l_-1.00_0.00_0.00.png"
            if not exists(join(self.root_dir, left_name)):
                logging.error("Missing left file: " + left_name)
                continue
            right_name = "ss1_" + seed + "r_1.00_0.00_0.00.png"
            if not exists(join(self.root_dir, right_name)):
                logging.error("Missing right file: " + right_name)
                continue
            position = np.array([float(fg.group(3)), float(fg.group(4)), float(fg.group(5))], dtype=np.float32)
            self.scene_data.append((f, left_name, right_name, position))
        logging.info("Done.")
        self.transform = transform

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, idx):
        generated_name = join(self.root_dir, self.scene_data[idx][0])
        left_name = join(self.root_dir, self.scene_data[idx][1])
        right_name = join(self.root_dir, self.scene_data[idx][2])
        position = self.scene_data[idx][3]
        scale = (1.0/255.0)
        generated_image = np.clip(scale * np.array(cv2.imread(generated_name), dtype=np.float32), 0.0, 1.0)[:,:,::-1]
        left_image = np.clip(scale * np.array(cv2.imread(left_name), dtype=np.float32), 0.0, 1.0)[:,:,::-1]
        right_image = np.clip(scale * np.array(cv2.imread(right_name), dtype=np.float32), 0.0, 1.0)[:,:,::-1]
        sample = {
            'generated_name': generated_name,
            'generated': generated_image,
            'left_name': left_name,
            'left': left_image,
            'right_name': right_name,
            'right': right_image,
            'position': position
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def main():
    # dataset_transforms = transforms.Compose([ToTensor()])
    dataset_transforms = transforms.Compose([SubsampleImages(0.125),
                                             ToTensor(),
                                             NormalizeImages(
                                                 mean=[0.0,0.0,0.0],
                                                 std=[1.0,1.0,1.0])
                                             ])
    dataset = RandomSceneDataset(root_dir="screens_256",
                                 transform=dataset_transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        sample_generated_names, sample_positions = sample_batched['generated_name'], sample_batched['position']
        generated_batch = sample_batched['generated']
        left_batch = sample_batched['left']
        right_batch = sample_batched['right']
        # expanded_position = sample_batched['expanded_position']
        # combined = torch.cat((generated_batch, left_batch, right_batch, expanded_position), dim=1)
        for_display = torch.cat((generated_batch, left_batch, right_batch), dim=2)

        # for i in range(batch_size):
        #     print(str(i_batch) + "." + str(i) + ": " + str(sample_generated_names[i]) +
        #     " , (" + str(sample_positions[:][i]))

        if i_batch == 3:
            plt.figure()
            grid = utils.make_grid(for_display)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
