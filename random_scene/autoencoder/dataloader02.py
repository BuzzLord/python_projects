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


# Old Dataloader for Siren files
class RandomSceneSirenFileList(Dataset):
    """Old (deprecated )Random scene dataset."""

    def __init__(self, root_dir, dataset_seed, batch_size, num_workers, pin_memory, test_percent=0.1, is_test=False,
                 shuffle=True, pos_scale=None, importance=None, transform=None):
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
        self.test_percent = min(max(test_percent, 0.0), 1.0)
        self.is_test = is_test
        self.file_names = []

        logging.info("Checking dataset directory listing for {}".format(root_dir))
        file_list = []
        for d in self.root_dirs:
            file_list.extend([join(d, f) for f in listdir(d) if isfile(join(d, f)) and f.startswith("ss") and f.endswith(".png")])
        logging.info("Found " + str(len(file_list)) + " files. Collating filenames into list.")
        all_file_names = []
        for f in file_list:
            fg = match("ss([34])_([0-9]*)_.*.png", basename(f))
            seed = fg.group(2)
            if seed != dataset_seed:
                continue

            version = fg.group(1)
            if version not in ["3", "4"]:
                logging.error("Skipping file, version not in [3,4]: " + f)
                continue

            all_file_names.append(f)

        random.seed(int(self.dataset_seed))
        random.shuffle(all_file_names)
        file_count = int(len(all_file_names) * self.test_percent)
        if self.is_test:
            self.file_names = all_file_names[0:file_count]
        else:
            self.file_names = all_file_names[file_count:]

        logging.info("Resulting file list contains " + str(len(self.file_names)) + " files.")

        self.transform = transform

    def generate_dataloader(self, file_list):
        sampleset = RandomSceneSirenSampleSetList(file_list=file_list,
                                                  pos_scale=self.pos_scale, importance=self.importance,
                                                  transform=self.transform)
        dataloader = torch.utils.data.DataLoader(sampleset, batch_size=self.batch_size, num_workers=self.num_workers,
                                                 pin_memory=self.pin_memory, shuffle=self.shuffle)
        return dataloader

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        return self.file_names[idx]


# Old Dataloader for sampling Siren images
class RandomSceneSirenSampleSetList(Dataset):
    """Old (Deprecated) Random scene dataset. """

    def __init__(self, file_list, pos_scale=None, transform=None, importance=None):
        """
        Args:
            file_list (list of strings): File list for the image to samples.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if pos_scale is None:
            pos_scale = [1.0, 1.0, 1.0]
        self.file_list = file_list
        logging.debug("Generating file samples from {} files ({}, ...)".format(len(file_list), file_list[0]))
        self.image = []
        self.inputs = []
        self.dims = []
        self.indices = torch.zeros((0,3), dtype=torch.int32)

        for i, file_path in enumerate(file_list):
            vg = match("ss([1234])_([0-9]*)_.*.png", basename(file_path))
            if vg.group(1) == "3":
                fg = match("ss([1234])_([0-9]*)_([0-9.-]*)_([0-9.-]*)_([0-9.-]*)_([lrg]).png", basename(file_path))
                pos_group = (float(fg.group(3)), float(fg.group(4)), float(fg.group(5)))
                rot_group = (0.0, 0.0)
            elif vg.group(1) == "4":
                fg = match("ss([1234])_([0-9]*)_([0-9.+-]*)_([0-9.+-]*)_([0-9.+-]*)_([0-9.+-]*)_([0-9.+-]*)_([lrg]).png", basename(file_path))
                pos_group = (float(fg.group(3)), float(fg.group(4)), float(fg.group(5)))
                rot_group = (float(fg.group(6)), float(fg.group(7)))
            else:
                logging.error("Invalid version {}".format(vg.group(1)))
                continue

            image = np.clip((1.0 / 255.0) * np.array(cv2.imread(file_path, cv2.IMREAD_UNCHANGED), dtype=np.float32), 0.0, 1.0)

            if importance:
                edges = cv2.Laplacian(cv2.cvtColor(image[:, :, 0:3], cv2.COLOR_BGR2GRAY), cv2.CV_32F, ksize=1)
                edges = importance * torch.abs(torch.from_numpy(edges))
            else:
                edges = torch.zeros((image.shape[0], image.shape[1]), dtype=torch.float32)

            index_count = torch.pow(2.0, edges).round_().view((image.shape[0]*image.shape[1])).type(torch.long)
            index_img = len(self.image) * torch.ones((image.shape[0], image.shape[1]), dtype=torch.int32)
            index_x = torch.linspace(0, image.shape[0]-1, image.shape[0], dtype=torch.int32)
            index_x = (index_x.unsqueeze(1) * torch.ones(image.shape[0], dtype=torch.int32)).transpose(dim0=0, dim1=1)
            index_y = torch.linspace(0, image.shape[1]-1, image.shape[1], dtype=torch.int32)
            index_y = (index_y.unsqueeze(1) * torch.ones(image.shape[1], dtype=torch.int32))
            index_stack = torch.stack((index_img, index_x, index_y), dim=2)
            index_stack = index_stack.view((image.shape[0]*image.shape[1], 3))
            index_expanded = torch.repeat_interleave(index_stack, index_count, dim=0)
            self.indices = torch.cat((self.indices, index_expanded), dim=0)

            image = 2.0 * image - 1.0
            image = image[:, :, 0:3][:, :, ::-1]
            self.image.append(torch.from_numpy(image.copy()))
            self.dims.append((image.shape[0], image.shape[1]))

            pos = torch.ones(image.shape, dtype=torch.float32)
            pos[:,:,0].mul_(pos_group[0] * pos_scale[0])
            pos[:,:,1].mul_(pos_group[1] * pos_scale[1])
            pos[:,:,2].mul_(pos_group[2] * pos_scale[2])

            theta = torch.linspace(-1.0 + (1/float(image.shape[0])),
                                   1.0 - (1/float(image.shape[0])), image.shape[0], dtype=torch.float32)
            theta = theta + rot_group[0] / 90
            theta = (theta.unsqueeze(1) * torch.ones(image.shape[0])).transpose(dim0=0, dim1=1)
            phi = torch.linspace(1.0 - (1/float(image.shape[1])),
                                 -1.0 + (1/float(image.shape[1])), image.shape[1], dtype=torch.float32)
            phi = phi + rot_group[1] / 90
            phi = (phi.unsqueeze(1) * torch.ones(image.shape[1]))
            inputs = torch.cat((pos, theta.unsqueeze(2), phi.unsqueeze(2)), dim=2)
            self.inputs.append(inputs)

        self.transform = transform

    def _get_input(self, img, x, y):
        return self.inputs[img][x, y]

    def _get_output(self, img, x, y):
        return self.image[img][x, y]

    def get_in_order_sample(self, img=0):
        return {"inputs": self.inputs[img].view(self.inputs[img].shape[0]*self.inputs[img].shape[1], 5),
                "outputs": self.image[img].view(self.image[img].shape[0]*self.image[img].shape[1], 3),
                "dims": self.dims[img]}

    def __len__(self):
        # return sum([self.image[i].shape[0] * self.image[i].shape[1] for i in range(len(self.image))])
        return self.indices.shape[0]

    def getitem_old(self, idx):
        img_idx = idx
        img = 0
        for _ in range(len(self.image)):
            if img_idx < (self.image[img].shape[0]*self.image[img].shape[1]):
                break
            else:
                img_idx -= (self.image[img].shape[0]*self.image[img].shape[1])
                img += 1

        ix = int(img_idx / self.image[img].shape[0])
        iy = int(img_idx % self.image[img].shape[0])
        sample = {"inputs": self._get_input(img, ix, iy), "outputs": self._get_output(img, ix, iy), "dims": self.dims[img]}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __getitem__(self, idx):
        idx_vector = self.indices[idx]
        img, ix, iy = idx_vector[0], idx_vector[1], idx_vector[2]
        sample = {"inputs": self._get_input(img, ix, iy), "outputs": self._get_output(img, ix, iy),
                  "dims": self.dims[img]}
        if self.transform:
            sample = self.transform(sample)
        return sample



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    print("Not implemented")
