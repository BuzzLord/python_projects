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


# -- SIREN ------------------------

pi_2 = 1.5707963267948966192313216916398


def angle_to_vector(theta, phi):
    u = pi_2 * theta
    v = pi_2 * phi
    sin_u, cos_u = torch.sin(u), torch.cos(u)
    sin_v, cos_v = torch.sin(v), torch.cos(v)
    return torch.stack((sin_u * cos_v, sin_v, -cos_u * cos_v), dim=1)


def rotate_vector(vec, theta, phi):
    u = pi_2 * theta
    v = pi_2 * phi
    sin_u, cos_u = torch.sin(u), torch.cos(u)
    sin_v, cos_v = torch.sin(v), torch.cos(v)
    x = vec[:, 0] * cos_u + vec[:, 1] * sin_u * sin_v - vec[:, 2] * sin_u * cos_v
    y = vec[:, 1] * cos_v - vec[:, 2] * sin_v
    z = vec[:, 0] * sin_u + vec[:, 1] * cos_u * sin_v + vec[:, 2] * cos_u * cos_v
    return torch.stack((x, y, z), dim=1)


def vector_to_angle(look):
    phi = torch.asin(look[:,1]) / pi_2
    theta = torch.atan2(look[:,0], -look[:,2]) / pi_2
    return theta, phi


class RandomSceneSirenFileListLoader(Dataset):
    """Random scene dataset."""

    def __init__(self, root_dir, dataset_seed, batch_size, num_workers, pin_memory, test_percent=0.1, is_test=False,
                 shuffle=True, pos_scale=None, importance=None, transform=None, device=None):
        """
        Args:
            root_dir (string or list): Directory with all the images.
            dataset_seed (string): String value of the integer seed of the scene we want
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if isinstance(root_dir, list):
            self.root_dirs = root_dir
        else:
            self.root_dirs = [root_dir]
        self.dataset_seed = dataset_seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        if pos_scale is None:
            self.pos_scale = [1.0, 1.0, 1.0]
        else:
            self.pos_scale = pos_scale
        self.importance = importance
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.test_percent = min(max(test_percent, 0.0), 1.0)
        self.is_test = is_test
        self.file_names = []

        logging.info("Checking dataset directory listing for {}".format(root_dir))
        for d in self.root_dirs:
            file_list = [join(d, f) for f in listdir(d) if isfile(join(d, f)) and f.startswith("ss") and f.endswith(".png")]
            match_files = []
            for f in file_list:
                fg = match("ss([34])_([0-9]*)_.*.png", basename(f))
                seed = fg.group(2)
                if seed != dataset_seed:
                    continue

                version = fg.group(1)
                if version not in ["3", "4"]:
                    logging.error("Skipping file, version not in [3,4]: " + f)
                    continue

                match_files.append(f)

            random.seed(int(self.dataset_seed))
            random.shuffle(match_files)
            file_count = round(len(match_files) * self.test_percent)
            if self.is_test:
                self.file_names.extend(match_files[0:file_count])
            else:
                self.file_names.extend(match_files[file_count:])

        logging.info("Resulting file list contains " + str(len(self.file_names)) + " files.")

        self.transform = transform

    def collate_fn(self, batch):
        return batch

    def generate_dataloader(self, sample_list, apply_transform=False, max_t=1.0):
        dataloader = RandomSceneSirenSampleLoader(sample_list=sample_list, batch_size=self.batch_size,
                                                  device=self.device, shuffle=self.shuffle,
                                                  apply_transform=apply_transform, max_t=max_t)
        return dataloader

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        vg = match("ss([1234])_([0-9]*)_.*.png", basename(self.file_names[idx]))
        if vg.group(1) == "3":
            fg = match("ss([1234])_([0-9]*)_([0-9.-]*)_([0-9.-]*)_([0-9.-]*)_([lrg]).png",
                       basename(self.file_names[idx]))
            pos_group = (float(fg.group(3)), float(fg.group(4)), float(fg.group(5)))
            rot_group = (0.0, 0.0)
        elif vg.group(1) == "4":
            fg = match("ss([1234])_([0-9]*)_([0-9.+-]*)_([0-9.+-]*)_([0-9.+-]*)_([0-9e.+-]*)_([0-9e.+-]*)_([lrg]).png",
                       basename(self.file_names[idx]))
            pos_group = (float(fg.group(3)), float(fg.group(4)), float(fg.group(5)))
            rot_group = (float(fg.group(6)), float(fg.group(7)))
        else:
            raise RuntimeError("Invalid version {}".format(vg.group(1)))

        image = np.clip((1.0 / 255.0) * np.array(cv2.imread(self.file_names[idx], cv2.IMREAD_UNCHANGED),
                                                 dtype=np.float32), 0.0, 1.0)
        edges = self.get_image_edges(image)
        sample_count = torch.pow(2.0, edges).round_().view((image.shape[0] * image.shape[1])).type(torch.long)
        sample_count = sample_count.to(self.device, dtype=torch.long)

        image = 2.0 * image - 1.0
        image = image[:, :, 0:3][:, :, ::-1]
        image_cuda = torch.from_numpy(image.copy()).to(self.device, dtype=torch.float32)

        position = self.construct_positions(image.shape[0], image.shape[1], pos_group)
        rotation = self.construct_rotation_vector(image.shape[0], image.shape[1], rot_group)

        inputs = torch.cat((position, rotation), dim=2)

        return { "filename": self.file_names[idx],
                 "image": image_cuda,
                 "inputs": inputs,
                 "sample_count": sample_count,
                 "dims": (image.shape[0], image.shape[1])
                 }

    def construct_rotation_vector(self, dim0, dim1, rot_group):
        theta_base = torch.linspace(-1.0 + (1 / float(dim0)), 1.0 - (1 / float(dim0)), dim0,
                                    device=self.device, dtype=torch.float32).repeat((dim1, 1))
        phi_base = torch.linspace(1.0 - (1 / float(dim1)), -1.0 + (1 / float(dim1)), dim1,
                                  device=self.device, dtype=torch.float32).repeat((dim0, 1)).transpose(1, 0)
        vector = angle_to_vector(theta_base.reshape(dim0 * dim1), phi_base.reshape(dim0 * dim1))
        rotation = torch.tensor([[rot_group[0] / 90.0, rot_group[1] / 90.0]],
                                device=self.device, dtype=torch.float32).repeat((vector.shape[0], 1))
        rotated_vector = rotate_vector(vector, rotation[:, 0], rotation[:, 1])
        return rotated_vector.reshape(dim0, dim1, 3)

    def construct_positions(self, dim0, dim1, pos_group):
        position = torch.ones((dim0, dim1, 3), device=self.device, dtype=torch.float32)
        position[:, :, 0].mul_(pos_group[0] * self.pos_scale[0])
        position[:, :, 1].mul_(pos_group[1] * self.pos_scale[1])
        position[:, :, 2].mul_(pos_group[2] * self.pos_scale[2])
        return position

    def get_image_edges(self, image):
        if self.importance:
            edges = cv2.Laplacian(cv2.cvtColor(image[:, :, 0:3], cv2.COLOR_BGR2GRAY), cv2.CV_32F, ksize=1)
            res_multiplier = np.min(np.log2(image.shape[0] * image.shape[1]) * 0.5 - 7, 0)
            edges = self.importance * res_multiplier * torch.abs(torch.from_numpy(edges))
        else:
            edges = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.float32)
        return edges


class RandomSceneSirenSampleLoader:
    """Random scene sample dataloader.
       Note: not a dataset to be used with the dataloader. It is itself a dataloader.
       Implements iterable.
    """

    def __init__(self, sample_list, batch_size, shuffle=True, device=None, apply_transform=True, max_t=1.0):
        """
        Args:
            sample_list (list(dict)): Sample list of images/inputs to sample pixels from.
        """
        self.sample_list = sample_list
        self.batch_size = batch_size
        self.shuffle = shuffle
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device
        self.apply_transform = apply_transform
        self.max_t = max_t

        self.dims = []

        index_list = []
        image_list = []
        inputs_list = []
        index_size = 0
        for i, sample in enumerate(sample_list):
            dims = sample["dims"]
            image_size = dims[0] * dims[1]

            index_linear = torch.linspace(index_size, index_size+image_size-1, image_size,
                                          device=self.device, dtype=torch.long)
            index_expanded = torch.repeat_interleave(index_linear, sample["sample_count"], dim=0)
            index_list.append(index_expanded)
            self.dims.append((dims[0], dims[1], index_size))
            index_size += image_size

            image = sample["image"]
            image = image.view((image_size, 3))
            image_list.append(image)

            inputs = sample["inputs"]
            inputs = inputs.view((image_size, 6))
            inputs_list.append(inputs)

        self.indices = torch.cat(index_list, dim=0)
        self.image = torch.cat(image_list, dim=0)
        self.inputs = torch.cat(inputs_list, dim=0)

        self.num_batches = int(np.ceil(self.indices.shape[0] / self.batch_size))
        if self.shuffle:
            shuffle_indices = torch.randperm(self.indices.shape[0], device=self.device, dtype=torch.long)
            self.indices = self.indices.index_select(0, shuffle_indices)

    def set_max_t(self, max_t):
        self.max_t = max_t

    def _get_input(self, img, x, y):
        return self.inputs[img][x, y]

    def _get_output(self, img, x, y):
        return self.image[img][x, y]

    def get_in_order_sample(self, img=0):
        dims = self.dims[img]
        img_size = dims[0]*dims[1]
        img_start = dims[2]
        return {"inputs": self.inputs[img_start:(img_start+img_size), :],
                "outputs": self.image[img_start:(img_start+img_size), :],
                "dims": self.dims[img]}

    def vector_transform(self, inputs):
        position = inputs[:, 0:3]
        look = inputs[:, 3:6]

        # Do a axis-aligned bounding box calculation to get nearest intersection in look dir, and behind.
        inv_look = 1 / look
        sign = (inv_look < 0.0).type(torch.float32) * 2.0 - 1.0
        tmin = (sign - position) * inv_look
        tmin_index = torch.abs(tmin).argmin(1)
        tmin_out = torch.clamp(torch.index_select(tmin, 1, tmin_index).diag(), -self.max_t, 0)
        tmax = (-sign - position) * inv_look
        tmax_index = torch.abs(tmax).argmin(1)
        tmax_out = torch.clamp(torch.index_select(tmax, 1, tmax_index).diag(), 0, self.max_t)

        # Randomly select a vector between the two intersection points, update pos to that
        t = (tmax_out - tmin_out) * torch.rand(tmax_out.shape, device=self.device) + tmin_out
        # max_position = position + look * tmax_out.unsqueeze(1)
        # min_position = position + look * tmin_out.unsqueeze(1)
        new_position = position + look * t.unsqueeze(1)
        inputs[:, 0:3] = new_position
        return inputs

    def __len__(self):
        return self.num_batches

    def __getitem__(self, idx):
        if idx >= self.num_batches:
            raise IndexError
        i = idx * self.batch_size
        j = min(i + self.batch_size, self.indices.shape[0])
        index_select = self.indices[i:j]
        image = self.image.index_select(0, index_select)
        inputs = self.inputs.index_select(0, index_select)
        if self.apply_transform:
            inputs = self.vector_transform(inputs)
        sample = {"inputs": inputs,
                  "outputs": image,
                  "dims": None
                  }
        return sample


def main():
    def deg2norm(angle):
        return angle / 90

    dims = (32, 32)
    sample = {"inputs": torch.zeros((dims[0], dims[1], 6), dtype=torch.float32),
              "image": torch.zeros((dims[0], dims[1], 3), dtype=torch.float32),
              "dims": (dims[0], dims[1]),
              "sample_count": torch.ones(dims[0]*dims[1], dtype=torch.long)}

    positions = torch.tensor([
        [-0.5,0,0.5],
        [0,0,0,],
        [0,-0.4330127,0.75],
        [0,0,0]], dtype=torch.float32)
    angles = torch.tensor([
        [deg2norm(45),0],
        [deg2norm(45), 0],
        [0,deg2norm(30)],
        [0,deg2norm(30)]], dtype=torch.float32)
    rotations = angle_to_vector(angles[:,0], angles[:,1])
    test_inputs = torch.cat((positions, rotations), dim=1)
    transform = RandomSceneSirenSampleLoader([sample], 1, max_t=10.0)
    transform.vector_transform(test_inputs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
