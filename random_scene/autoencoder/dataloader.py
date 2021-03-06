from os import listdir
from os.path import isfile, join, exists, basename
from re import match
import torch
import numpy as np
import random
import threading
import queue
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import cv2
import matplotlib.pyplot as plt
import logging
from projection import Projection




class ExpandPosition(object):
    """Expand the 3x1 position vector into a 3xHxW image/numpy array."""

    def __call__(self, sample):
        generated_image = sample['generated']
        position = sample['position']
        expanded_position = position * np.ones((generated_image.shape[0], generated_image.shape[1], position.size),
                                               dtype=np.float32)

        sample['expanded_position'] = expanded_position

        return sample


class SubsampleImages(object):
    """Downsample then upsample the images in the sample by scale."""

    def __call__(self, sample):
        def rescale(img, scale):
            orig_dim = (img.shape[1], img.shape[0])
            scale_dim = (int(orig_dim[0] * scale), int(orig_dim[1] * scale))
            if scale_dim == orig_dim:
                return img
            interp_a = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            interp_b = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_LINEAR
            return cv2.resize(
                cv2.resize(img, scale_dim, interpolation=interp_a),
                orig_dim, interpolation=interp_b)

        sample['generated'] = rescale(sample['generated'], 1.0)
        sample['left'] = rescale(sample['left'], 1.0)
        sample['right'] = rescale(sample['right'], 1.0)

        if 'generated_depth' in sample:
            sample['generated_depth'] = rescale(sample['generated_depth'], 1.0)

        return sample


class SubsampleInputImages(object):
    """Downsample then upsample the images in the sample by scale."""

    def __call__(self, sample):
        def rescale(img, scale):
            orig_dim = (img.shape[1], img.shape[0])
            scale_dim = (int(orig_dim[0] * scale), int(orig_dim[1] * scale))
            if scale_dim == orig_dim:
                return img
            interp_a = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            interp_b = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_LINEAR
            return cv2.resize(
                cv2.resize(img, scale_dim, interpolation=interp_a),
                orig_dim, interpolation=interp_b)

        if sample['super_res_factor'] > 1.0 and sample['resample'] > 1.0:
            sample['left'] = rescale(sample['left'], 1.0 / sample['super_res_factor'])
            sample['right'] = rescale(sample['right'], 1.0 / sample['super_res_factor'])

        return sample


class SubsampleGeneratedImages(object):
    """Downsample then upsample the images in the sample by scale."""

    def __call__(self, sample):
        def rescale(img, scale):
            orig_dim = (img.shape[1], img.shape[0])
            scale_dim = (int(orig_dim[0] * scale), int(orig_dim[1] * scale))
            if scale_dim == orig_dim:
                return img
            interp_a = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            interp_b = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_LINEAR
            return cv2.resize(
                cv2.resize(img, scale_dim, interpolation=interp_a),
                orig_dim, interpolation=interp_b)

        sample['generated'] = rescale(sample['generated'], 1.0)

        return sample


class SubsampleGeneratedDepth(object):
    """Downsample then upsample the images in the sample by scale."""

    def __call__(self, sample):
        def rescale(img, scale):
            orig_dim = (img.shape[1], img.shape[0])
            scale_dim = (int(orig_dim[0] * scale), int(orig_dim[1] * scale))
            if scale_dim == orig_dim:
                return img
            interp_a = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            interp_b = cv2.INTER_AREA if scale > 1.0 else cv2.INTER_LINEAR
            return cv2.resize(
                cv2.resize(img, scale_dim, interpolation=interp_a),
                orig_dim, interpolation=interp_b)

        if 'generated_depth' in sample:
            sample['generated_depth'] = rescale(sample['generated_depth'], 1.0)

        return sample


class ResampleImages(object):
    """Downsample or upsample the images in the sample by scale."""

    def __call__(self, sample):
        def rescale(img, scale):
            scale_dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            if scale_dim == (img.shape[1], img.shape[0]):
                return img
            interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            return cv2.resize(img, scale_dim, interpolation=interpolation)

        sample['generated'] = rescale(sample['generated'], sample['resample'])
        sample['left'] = rescale(sample['left'], sample['resample'] / sample['super_res_factor'])
        sample['right'] = rescale(sample['right'], sample['resample'] / sample['super_res_factor'])

        if 'generated_depth' in sample:
            sample['generated_depth'] = rescale(sample['generated_depth'], sample['resample'])

        return sample


class ResampleInputImages(object):
    """Downsample or upsample the input images in the sample by scale."""

    def __call__(self, sample):
        def rescale(img, scale):
            scale_dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            if scale_dim == (img.shape[1], img.shape[0]):
                return img
            interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            return cv2.resize(img, scale_dim, interpolation=interpolation)

        sample['left'] = rescale(sample['left'], 1.0 / sample['super_res_factor'])
        sample['left'] = rescale(sample['left'], sample['resample'])
        sample['right'] = rescale(sample['right'], 1.0 / sample['super_res_factor'])
        sample['right'] = rescale(sample['right'], sample['resample'])

        return sample


class ResampleGeneratedImages(object):
    """Downsample or upsample the images in the sample by scale."""

    def __call__(self, sample):
        def rescale(img, scale):
            scale_dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            if scale_dim == (img.shape[1], img.shape[0]):
                return img
            interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            return cv2.resize(img, scale_dim, interpolation=interpolation)

        sample['generated'] = rescale(sample['generated'], sample['resample'])

        return sample


class ResampleGeneratedDepth(object):
    """Downsample or upsample the images in the sample by scale."""

    def __call__(self, sample):
        def rescale(img, scale):
            scale_dim = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            if scale_dim == (img.shape[1], img.shape[0]):
                return img
            interpolation = cv2.INTER_LINEAR if scale > 1.0 else cv2.INTER_AREA
            return cv2.resize(img, scale_dim, interpolation=interpolation)

        if 'generated_depth' in sample:
            sample['generated_depth'] = rescale(sample['generated_depth'], sample['resample'])

        return sample


class GenerateReprojectedImages(object):
    def __init__(self, name, scale, xrot=0.0, yrot=0.0, fov=90.0, aspect=1.0):
        self.name = "reproj_" + name
        self.scale = scale
        self.proj = Projection(xrot=xrot, yrot=yrot, fov=fov, aspect=aspect)

    def __call__(self, sample):
        input_dim = (sample['generated'].shape[1], sample['generated'].shape[0])
        output_dim = (input_dim[0] * self.scale, input_dim[1] * self.scale)
        phi, theta = self.proj.generate_map(output_dim=output_dim, input_dim=input_dim)
        interpolation = cv2.INTER_AREA
        remap = cv2.remap(sample['generated'], phi, theta, interpolation=interpolation, borderValue=0,
                          borderMode=cv2.BORDER_CONSTANT)
        sample[self.name] = remap

        return sample


class UnwarpImages(object):
    def __init__(self, scale=1.0, fov=97.62815):
        # 106.2602 comes from 2*atan(4/3), to give the 90 deg inner image a dim of 3/4 of the full dim.
        # 97.62815 comes from 2*atan(8/7), for 7/8 image as inner 90 deg
        self.scale = scale
        self.proj = {
            'forward': Projection(xrot=0.0, yrot=0.0, fov=fov, aspect=1.0),
            'left': Projection(xrot=0.0, yrot=-90.0, fov=fov, aspect=1.0),
            'right': Projection(xrot=0.0, yrot=90.0, fov=fov, aspect=1.0),
            'up': Projection(xrot=-90.0, yrot=0.0, fov=fov, aspect=1.0),
            'down': Projection(xrot=90.0, yrot=0.0, fov=fov, aspect=1.0)
        }

    def _remap_image(self, sample, image):
        input_dim = (sample[image].shape[1], sample[image].shape[0])
        output_dim = (input_dim[0] * self.scale * 0.5, input_dim[1] * self.scale * 0.5)
        output_shape = (int(sample[image].shape[0] * self.scale), int(sample[image].shape[1] * self.scale), sample[image].shape[2])
        interpolation = cv2.INTER_AREA
        remap = {}
        for dir in self.proj.keys():
            phi, theta = self.proj[dir].generate_map(output_dim=output_dim, input_dim=input_dim)
            remap[dir] = cv2.remap(sample[image], phi, theta, interpolation=interpolation, borderValue=0, borderMode=cv2.BORDER_CONSTANT)

        output_img = np.zeros(output_shape)
        # output_img[0:int(output_shape[0] / 4), int(output_shape[1] / 4):int(output_shape[1] * 3 / 4), :] = \
        #     remap['left'][int(output_dim[0] / 2):, :, :]
        # output_img[int(output_shape[0] / 4):int(output_shape[0] * 3 / 4), int(output_shape[1] / 4):int(output_shape[1] * 3 / 4), :] = \
        #     remap['forward'][:, :, :]
        # output_img[int(output_shape[0] * 3 / 4):, int(output_shape[1] / 4):int(output_shape[1] * 3 / 4), :] = \
        #     remap['right'][0:int(output_dim[0] / 2), :, :]
        # output_img[int(output_shape[0] / 4):int(output_shape[0] * 3 / 4), 0:int(output_shape[1] / 4), :] = \
        #     remap['up'][:, int(output_dim[1] / 2):, :]
        # output_img[int(output_shape[0] / 4):int(output_shape[0] * 3 / 4), int(output_shape[1] * 3 / 4):, :] = \
        #     remap['down'][:, 0:int(output_dim[1] / 2), :]

        output_img[int(output_shape[0] / 4):int(output_shape[0] * 3 / 4), int(output_shape[1] / 4):int(output_shape[1] * 3 / 4), :] = \
            remap['forward'][:, :, :]
        output_img[int(output_shape[0] / 4):int(output_shape[0] * 3 / 4), int(output_shape[1] * 3 / 4):, :] = \
            remap['up'][:, 0:int(output_dim[1] / 2), :]
        output_img[int(output_shape[0] / 4):int(output_shape[0] * 3 / 4), 0:int(output_shape[1] / 4), :] = \
            remap['down'][:, int(output_dim[1] / 2):, :]
        output_img[int(output_shape[0] * 3 / 4):, int(output_shape[1] / 4):int(output_shape[1] * 3 / 4), :] = \
            remap['left'][0:int(output_dim[0] / 2), :, :]
        output_img[0:int(output_shape[0] / 4), int(output_shape[1] / 4):int(output_shape[1] * 3 / 4), :] = \
            remap['right'][int(output_dim[0] / 2):, :, :]

        sample[image] = output_img

    def __call__(self, sample):
        self._remap_image(sample, 'generated')
        self._remap_image(sample, 'left')
        self._remap_image(sample, 'right')

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # print(str(dt.now()) + " ToTensoring " + sample['generated_name'])

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        sample['generated'] = torch.from_numpy(sample['generated'].transpose((2, 0, 1)))
        sample['left'] = torch.from_numpy(sample['left'].transpose((2, 0, 1)))
        sample['right'] = torch.from_numpy(sample['right'].transpose((2, 0, 1)))
        sample['position'] = torch.reshape(torch.from_numpy(sample['position']), (3, 1, 1))

        if 'expanded_position' in sample:
            sample['expanded_position'] = torch.from_numpy(sample['expanded_position'].transpose((2, 0, 1)))

        if 'generated_depth' in sample:
            sample['generated_depth'] = torch.from_numpy(np.expand_dims(sample['generated_depth'], axis=2).transpose((2, 0, 1)))

        for n in sample:
            if n.startswith("reproj_"):
                sample[n] = torch.from_numpy(sample[n].transpose((2, 0, 1)))

        return sample


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
        generated_image = sample['generated']
        left_image, right_image = sample['left'], sample['right']

        for t, m, s in zip(generated_image, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(left_image, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(right_image, self.mean, self.std):
            t.sub_(m).div_(s)
        # print(str(dt.now()) + " Finished " + sample['generated_name'])

        sample['generated'] = generated_image
        sample['left'] = left_image
        sample['right'] = right_image

        for name in sample:
            if name.startswith("reproj_"):
                reproj = sample[name]
                for t, m, s in zip(reproj, self.mean, self.std):
                    t.sub_(m).div_(s)
                sample[name] = reproj

        return sample


class NormalizeDepth(object):
    """Normalize a tensor depth images with mean and standard deviation.
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
            Tensor: Normalized Tensor depth image.
        """
        if 'generated_depth' not in sample:
            return sample

        generated_image_depth = sample['generated_depth']

        for t, m, s in zip(generated_image_depth, self.mean, self.std):
            t.sub_(m).div_(s)

        sample['generated_depth'] = generated_image_depth

        return sample


class RandomSceneDataset(Dataset):
    """Random scene dataset."""

    def __init__(self, root_dir, transform=None, depth=False, reverse=False, super_res_factor=1.0, resample_rate=1.0):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.scene_data = []
        self.depth = depth
        self.super_res_factor = super_res_factor
        self.resample_rate = resample_rate
        logging.info("Checking dataset directory listing for {}".format(root_dir))
        file_list = [f for f in listdir(self.root_dir) if isfile(join(self.root_dir, f)) and f.startswith("ss") and f.endswith(".png")]
        logging.info("Found " + str(len(file_list)) + " files. Collating filenames into list.")
        for f in file_list:
            fg = match("ss([12])_([0-9]*)([lrg])_([0-9.\-]*)_([0-9.\-]*)_([0-9.\-]*)\.png", f)
            seed = fg.group(2)
            version = fg.group(1)
            if depth:
                if version not in ["2"]:
                    logging.error("Skipping file, version != 2: " + f)
                    continue
            else:
                if version not in ["1"]:
                    logging.error("Skipping file, version != 1: " + f)
                    continue

            left_name = "ss" + version + "_" + seed + "l_-1.00_0.00_0.00.png"
            if not exists(join(self.root_dir, left_name)):
                logging.error("Missing left file: " + left_name)
                continue
            right_name = "ss" + version + "_" + seed + "r_1.00_0.00_0.00.png"
            if not exists(join(self.root_dir, right_name)):
                logging.error("Missing right file: " + right_name)
                continue
            position = np.array([float(fg.group(4)), float(fg.group(5)), float(fg.group(6))], dtype=np.float32)

            self.scene_data.append((f, left_name, right_name, position, False))
            if reverse:
                self.scene_data.append((f, left_name, right_name, position, True))
        self.transform = transform

    def __len__(self):
        return len(self.scene_data)

    def __getitem__(self, idx):
        generated_name = join(self.root_dir, self.scene_data[idx][0])
        left_name = join(self.root_dir, self.scene_data[idx][1])
        right_name = join(self.root_dir, self.scene_data[idx][2])
        position = self.scene_data[idx][3]
        is_reversed = self.scene_data[idx][4]
        scale = (1.0/255.0)
        generated_image = np.clip(scale * np.array(cv2.imread(generated_name, cv2.IMREAD_UNCHANGED), dtype=np.float32), 0.0, 1.0)
        left_image = np.clip(scale * np.array(cv2.imread(left_name, cv2.IMREAD_UNCHANGED), dtype=np.float32), 0.0, 1.0)
        right_image = np.clip(scale * np.array(cv2.imread(right_name, cv2.IMREAD_UNCHANGED), dtype=np.float32), 0.0, 1.0)
        if is_reversed:
            generated_image = cv2.flip(generated_image, 1)
            temp_image = cv2.flip(left_image, 1)
            left_image = cv2.flip(right_image, 1)
            right_image = temp_image
            position[0] = -position[0]

        generated_image_color = np.copy(generated_image[:, :, 0:3][:, :, ::-1])
        left_image_color = np.copy(left_image[:, :, 0:3][:, :, ::-1])
        right_image_color = np.copy(right_image[:, :, 0:3][:, :, ::-1])

        sample = {
            'generated_name': generated_name,
            'generated': generated_image_color,
            'left_name': left_name,
            'left': left_image_color,
            'right_name': right_name,
            'right': right_image_color,
            'position': position,
            'resample': self.resample_rate,
            'super_res_factor': self.super_res_factor
        }

        if self.depth:
            sample['generated_depth'] = generated_image[:, :, 3]

        if self.transform:
            sample = self.transform(sample)

        return sample


def main():
    # dataset_transforms = transforms.Compose([ToTensor()])
    dataset_transforms = transforms.Compose([UnwarpImages(scale=1.5, fov=110.0),
                                             ToTensor(),
                                             NormalizeImages(
                                                 mean=[0.0,0.0,0.0],
                                                 std=[1.0,1.0,1.0])
                                             ])
    dataset = RandomSceneDataset(root_dir="test_256",
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
