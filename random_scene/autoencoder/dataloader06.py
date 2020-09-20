from os import listdir
from os.path import isfile, join, exists, basename, getsize
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

    def __init__(self, root_dir, dataset_seed, batch_size, num_workers, pin_memory, img_set_size=1048576,
                 test_percent=0.1, is_test=False, shuffle=True, pos_scale=None, importance=None, transform=None):
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
        self.epsilon = 0.0
        if pos_scale is None:
            self.pos_scale = [1.0, 1.0, 1.0]
        else:
            self.pos_scale = pos_scale
        self.importance = importance
        self.test_percent = min(max(test_percent, 0.0), 1.0)
        self.is_test = is_test
        self.file_names = []
        self.read_files_from_root_dirs(dataset_seed, root_dir)

        logging.info("Resulting file list contains {} files.".format(len(self.file_names)))

        self.img_sets = []
        self.img_set_size = img_set_size
        self.construct_img_sets()

        self.transform = transform

    def read_files_from_root_dirs(self, dataset_seed, root_dir):
        logging.info("Checking dataset directory listing for {}".format(root_dir))
        for d in self.root_dirs:
            file_list = [join(d, f) for f in listdir(d) if
                         isfile(join(d, f)) and f.startswith("ss") and f.endswith(".png")]
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

    def construct_img_sets(self):
        random.shuffle(self.file_names)
        next_batch = []
        next_batch_size = 0
        for f in self.file_names:
            try:
                f_size = getsize(f)
            except OSError:
                logging.error("Got an OS Error when checking {}".format(f))
                continue
            if (next_batch_size + f_size) > self.img_set_size:
                if len(next_batch) == 0:
                    logging.warning("Single file '{}' is over img_set_size ({})".format(f, self.img_set_size))
                    next_batch.append(f)
                    self.img_sets.append(next_batch)
                    next_batch = []
                    next_batch_size = 0
                else:
                    self.img_sets.append(next_batch)
                    next_batch = [f]
                    next_batch_size = f_size
            else:
                next_batch.append(f)
                next_batch_size += f_size
        if len(next_batch) > 0:
            self.img_sets.append(next_batch)

    def collate_fn(self, batch):
        return [sample for sublist in batch for sample in sublist]

    def generate_dataloader(self, sample_list, apply_transform=False, max_t=1.0):
        dataloader = RandomSceneSirenSampleLoader(sample_list=sample_list, batch_size=self.batch_size,
                                                  shuffle=self.shuffle, apply_transform=apply_transform, max_t=max_t)
        return dataloader

    def __len__(self):
        return len(self.img_sets)

    def __getitem__(self, idx):
        img_set = self.img_sets[idx]
        samples = []
        for file_name in img_set:
            vg = match("ss([1234])_([0-9]*)_.*.png", basename(file_name))
            if vg.group(1) == "3":
                fg = match("ss([1234])_([0-9]*)_([0-9.-]*)_([0-9.-]*)_([0-9.-]*)_([lrg]).png",
                           basename(file_name))
                pos_group = (float(fg.group(3)), float(fg.group(4)), float(fg.group(5)))
                rot_group = (0.0, 0.0)
            elif vg.group(1) == "4":
                fg = match("ss([1234])_([0-9]*)_([0-9.+-]*)_([0-9.+-]*)_([0-9.+-]*)_([0-9e.+-]*)_([0-9e.+-]*)_([lrg]).png",
                           basename(file_name))
                pos_group = (float(fg.group(3)), float(fg.group(4)), float(fg.group(5)))
                rot_group = (float(fg.group(6)), float(fg.group(7)))
            else:
                raise RuntimeError("Invalid version {}".format(vg.group(1)))

            image = np.clip((1.0 / 255.0) * np.array(cv2.imread(file_name, cv2.IMREAD_UNCHANGED),
                                                     dtype=np.float32), 0.0, 1.0)
            edges = self.get_image_edges(image)

            image = 2.0 * image - 1.0
            image = image[:, :, 0:3][:, :, ::-1]
            image_cuda = torch.from_numpy(image.copy()).cuda(non_blocking=True)

            position = self.construct_positions(image.shape[0], image.shape[1], pos_group)
            theta, phi, angles_valid = self.construct_angles(image.shape[0], image.shape[1], rot_group)

            sample_count = torch.pow(2.0, edges).round().type(torch.long) * angles_valid
            sample_count = sample_count.view((image.shape[0] * image.shape[1])).cuda(non_blocking=True)

            inputs = torch.cat((position, theta, phi), dim=2).cuda(non_blocking=True)

            samples.append({"filename": file_name,
                            "image": image_cuda,
                            "inputs": inputs,
                            "sample_count": sample_count,
                            "dims": (image.shape[0], image.shape[1])
                            })
        return samples

    def construct_angles(self, dim0, dim1, rot_group):
        theta_base = torch.linspace(-1.0 + (1 / float(dim0)), 1.0 - (1 / float(dim0)), dim0,
                                    dtype=torch.float32).repeat((dim1, 1))
        phi_base = torch.linspace(1.0 - (1 / float(dim1)), -1.0 + (1 / float(dim1)), dim1,
                                  dtype=torch.float32).repeat((dim0, 1)).transpose(1, 0)
        vector = angle_to_vector(theta_base.reshape(dim0 * dim1), phi_base.reshape(dim0 * dim1))
        rotation = torch.tensor([[rot_group[0] / 90.0, rot_group[1] / 90.0]],
                                dtype=torch.float32).repeat((vector.shape[0], 1))
        rotated_vector = rotate_vector(vector, rotation[:, 0], rotation[:, 1])
        angles_valid = (rotated_vector[:, 2] < self.epsilon).type(torch.long)
        theta, phi = vector_to_angle(rotated_vector)
        return theta.reshape(dim0, dim1, 1), phi.reshape(dim0, dim1, 1), angles_valid.reshape(dim0, dim1)

    def construct_positions(self, dim0, dim1, pos_group):
        position = torch.ones((dim0, dim1, 3), dtype=torch.float32)
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

    def __init__(self, sample_list, batch_size, shuffle=True, apply_transform=True, max_t=1.0):
        """
        Args:
            sample_list (list(dict)): Sample list of images/inputs to sample pixels from.
        """
        self.sample_list = sample_list
        self.batch_size = batch_size
        self.shuffle = shuffle
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
                                          device=sample["sample_count"].device, dtype=torch.long)
            index_expanded = torch.repeat_interleave(index_linear, sample["sample_count"], dim=0)
            index_list.append(index_expanded)
            self.dims.append((dims[0], dims[1], index_size))
            index_size += image_size

            image = sample["image"]
            image = image.view((image_size, 3))
            image_list.append(image)

            inputs = sample["inputs"]
            inputs = inputs.view((image_size, 5))
            inputs_list.append(inputs)

        self.indices = torch.cat(index_list, dim=0)
        self.image = torch.cat(image_list, dim=0)
        self.inputs = torch.cat(inputs_list, dim=0)

        self.num_batches = int(np.ceil(self.indices.shape[0] / self.batch_size))
        if self.shuffle:
            shuffle_indices = torch.randperm(self.indices.shape[0], device=self.indices.device, dtype=torch.long)
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
        look = angle_to_vector(inputs[:, 3], inputs[:, 4])

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
        t = (tmax_out - tmin_out) * torch.rand(tmax_out.shape, device=inputs.device) + tmin_out
        max_position = position + look * tmax_out.unsqueeze(1)
        min_position = position + look * tmin_out.unsqueeze(1)
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


class SirenSampleRandomizePosition(object):
    def __init__(self, apply_transform=True, max_t=1.0):
        self.apply_transform = apply_transform
        self.max_t = max_t

    def set_max_t(self, max_t):
        self.max_t = max_t

    def vector_transform(self, inputs, device=None):
        if not self.apply_transform:
            return inputs
        if device is None:
            device = inputs.device
        position = inputs[:, 0:3]

        # Take sin values for theta/phi look angle, get cos of them
        u = 0.5 * np.pi * inputs[:, 3]
        v = 0.5 * np.pi * inputs[:, 4]
        sin_u, cos_u = torch.sin(u), torch.cos(u)
        sin_v, cos_v = torch.sin(v), torch.cos(v)

        # # Construct rotation matrices
        # zeros_u = torch.zeros(sin_u.shape, device=device)
        # zeros_v = torch.zeros(sin_v.shape, device=device)
        # ones_u = torch.ones(sin_u.shape, device=device)
        # ones_v = torch.ones(sin_v.shape, device=device)
        # rot_y = torch.stack((torch.stack((cos_u, zeros_u, sin_u), dim=1),
        #                      torch.stack((zeros_u, ones_u, zeros_u), dim=1),
        #                      torch.stack((-sin_u, zeros_u, cos_u), dim=1)), dim=-1)
        # rot_x = torch.stack((torch.stack((ones_v, zeros_v, zeros_v), dim=1),
        #                      torch.stack((zeros_v, cos_v, sin_v), dim=1),
        #                      torch.stack((zeros_v, -sin_v, cos_v), dim=1)), dim=-1)
        # rot = torch.matmul(rot_y, rot_x)
        #
        # # Rotate a forward vector by the rot matrices
        # vec = (torch.ones((sin_u.shape[0], 1), device=device) *
        #        torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=device)).unsqueeze(-1)
        # look = torch.matmul(rot, vec).squeeze(-1)

        look = torch.stack((sin_u * cos_v, sin_v, -cos_u * cos_v), dim=1)

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
        t = (tmax_out - tmin_out) * torch.rand(tmax_out.shape, device=device) + tmin_out
        # max_position = position + look * tmax_out.unsqueeze(1)
        # min_position = position + look * tmin_out.unsqueeze(1)
        new_position = position + look * t.unsqueeze(1)
        inputs[:, 0:3] = new_position
        return inputs

    def __call__(self, sample):
        inputs = sample["inputs"]
        position = inputs[0:3]

        # Take sin values for theta/phi look angle, get cos of them
        u = 0.5 * np.pi * inputs[3]
        v = 0.5 * np.pi * inputs[4]
        sin_u, cos_u = torch.sin(u), torch.cos(u)
        sin_v, cos_v = torch.sin(v), torch.cos(v)

        look = np.array([sin_u * cos_v, sin_v, -cos_u * cos_v], dtype=np.float32).transpose((1, 0))

        # Do a axis-aligned bounding box calculation to get nearest intersection in look dir, and behind.
        inv_look = 1 / look
        sign = (inv_look < 0.0).type(torch.float32) * 2.0 - 1.0
        tmin = (sign - position) * inv_look
        tmin_index = torch.abs(tmin).argmin(0)
        tmin_out = torch.clamp(torch.index_select(tmin, 0, tmin_index), -self.max_t, 0)
        tmax = (-sign - position) * inv_look
        tmax_index = torch.abs(tmax).argmin(0)
        tmax_out = torch.clamp(torch.index_select(tmax, 0, tmax_index), 0, self.max_t)
        # Randomly select a vector between the two intersection points, update pos to that
        t = (tmax_out - tmin_out) * torch.rand(tmax_out.shape) + tmin_out
        # max_position = position + look * tmax_out.unsqueeze(1)
        # min_position = position + look * tmin_out.unsqueeze(1)
        new_position = position + look * t.unsqueeze(1)
        inputs[0:3] = new_position
        sample["inputs"] = inputs
        return sample


def main_transform():
    def deg2norm(angle):
        return angle / 90

    dims = (32, 32)
    sample = {"inputs": torch.zeros((dims[0], dims[1], 5), dtype=torch.float32),
              "image": torch.zeros((dims[0], dims[1], 3), dtype=torch.float32),
              "dims": (dims[0], dims[1]),
              "sample_count": torch.ones(dims[0]*dims[1], dtype=torch.long)}

    test_inputs = torch.tensor([[-0.5,0,0.5,deg2norm(45),0],[0,0,0,deg2norm(45),0],
                               [0,-0.4330127,0.75,0,deg2norm(30)],[0,0,0,0,deg2norm(30)]], dtype=torch.float32)
    transform = RandomSceneSirenSampleLoader([sample], 1, max_t=10.0)
    transform.vector_transform(test_inputs)


def main():
    loader_test = RandomSceneSirenFileListLoader(root_dir=[], dataset_seed="1", batch_size=1, num_workers=1, pin_memory=True)

    res_angles = {
        2: (1 / 2, 3.918265e-1),
        4: (1 / 4, 1.000796e-1),
        8: (1 / 8, 2.469218e-2),
        16: (1 / 16, 6.145642e-3),
        32: (1 / 32, 1.534594e-3),
        64: (1 / 64, 3.835311e-4),
        128: (1 / 128, 9.587665e-5),
        256: (1 / 256, 2.396875e-5),
        512: (1 / 512, 5.992043e-6),
        1024: (1 / 1024, 1.497961e-6),
        2048: (1 / 2048, 3.744597e-7),
        4096: (1 / 4096, 9.361220e-8)
    }

    for res in res_angles.keys():
        angle_pairs = [(0,0), (res_angles[res][0], 0), (-res_angles[res][0], 0), (0, res_angles[res][1]),
                       (0, -res_angles[res][1]), (res_angles[res][0]*0.5, 0.5*res_angles[res][1])]
        for pair in angle_pairs:
            theta, phi = loader_test.construct_angles(res, res, (pair[0]*90, pair[1]*90))
            calc_angles = torch.tensor([[theta[0,0], phi[0,0]],
                                        [theta[-1,0],phi[-1,0]],
                                        [theta[0,-1], phi[0,-1]],
                                        [theta[-1,-1], phi[-1,-1]]])
            edge = 1.0 - res_angles[res][0]
            angles = torch.tensor([[-edge, edge],
                                   [edge, edge],
                                   [-edge,-edge],
                                   [edge,-edge]])
            looks_unrotated = angle_to_vector(angles[:,0], angles[:,1])
            rot_angles = torch.tensor([[pair[0], pair[1]]]).repeat((looks_unrotated.shape[0], 1))
            looks_rotated = rotate_vector(looks_unrotated, rot_angles[:, 0], rot_angles[:, 1])

            looks2 = angle_to_vector(calc_angles[:,0], calc_angles[:,1])
            mse = torch.sum(torch.pow(looks_rotated.flatten() - looks2.flatten(), 2))
            logging.info("MSE for {} {}: {}".format(res, pair, mse))


def main_search():
    def get_theta(vec, phi_angle):
        rot_angles = torch.tensor([[0.0, phi_angle]]).repeat((vec.shape[0], 1))
        rotated_looks = rotate_vector(vec, rot_angles[:, 0], rot_angles[:, 1])
        theta, phi = vector_to_angle(rotated_looks)
        return theta

    powers = {}
    for pwr in range(1, 13, 1):
        res = pow(2, pwr)
        # logging.info("Searching for {}".format(res))
        e = 1.0 - 1/res
        angle = torch.tensor([[e,e]])
        look = angle_to_vector(angle[:,0], angle[:,1])
        point_a = 0.5/res * pow(2, -(pwr-1))
        theta_a = get_theta(look, point_a)
        point_b = 1.0/res * pow(2, -(pwr-1))
        theta_b = get_theta(look, point_b)
        searching = True
        while searching:
            ratio = (1 - theta_b) / (theta_a - theta_b)
            point_x = ratio * point_a + (1 - ratio) * point_b
            theta_x = get_theta(look, point_x)
            if theta_x <= 1.0:
                if theta_x >= (1 - 1e-8):
                    powers[pwr] = point_x
                    searching = False
                else:
                    point_a = point_x
                    theta_a = theta_x
            else:
                if theta_x > theta_b:
                    logging.error("new theta greater than theta_b?")
                theta_b = theta_x
                point_b = point_x

        logging.info("{}: {:.6e}".format(res, powers[pwr][0]))


def main_angles():
    res_angles = {
        2: (1 / 2, 3.918265e-1),
        4: (1 / 4, 1.000796e-1),
        8: (1 / 8, 2.469218e-2),
        16: (1 / 16, 6.145642e-3),
        32: (1 / 32, 1.534594e-3),
        64: (1 / 64, 3.835311e-4),
        128: (1 / 128, 9.587665e-5),
        256: (1 / 256, 2.396875e-5),
        512: (1 / 512, 5.992043e-6),
        1024: (1 / 1024, 1.497961e-6),
        2048: (1 / 2048, 3.744597e-7),
        4096: (1 / 4096, 9.361220e-8)
    }

    for res in res_angles.keys():
        # res = 64
        edge = 1.0 - res_angles[res][0]
        angles = torch.tensor([[-edge, edge], [0.0, edge], [edge, edge],
                               [-edge, 0.0],  [0.0, 0.0],  [edge, 0.0],
                               [-edge,-edge], [0.0,-edge], [edge,-edge]])
        looks = angle_to_vector(angles[:,0], angles[:,1])

        combos = [i/10 for i in range(0, 11, 1)]
        # combos = [0]
        for u in combos:
            rot_angles = torch.tensor([[-res_angles[res][0] * u, (1.0 - u) * res_angles[res][1]]]).repeat((looks.shape[0],1))
            rotated_looks = rotate_vector(looks, rot_angles[:,0], rot_angles[:,1])
            theta, phi = vector_to_angle(rotated_looks)
            logging.info("For res {}, angle {:.8f}, {:.8f} => theta: {:.8f}, phi: {:.8f}".format(res, rot_angles[0][0],
                                                                                                 rot_angles[0][1],
                                                                                                 theta[0], phi[0]))
            if torch.min(theta) < -1.0:
                logging.error("This theta was too small!")
            if torch.max(theta) > 1.0:
                logging.error("This theta was too large!")
            if torch.min(phi) < -1.0:
                logging.error("This phi was too small!")
            if torch.max(phi) > 1.0:
                logging.error("This phi was too large!")

            # rotated_looks_from_angles = angle_to_vector(theta, phi)
            # rotated_looks_comparison = torch.mm(rotated_looks, rotated_looks_from_angles.transpose(1,0)).diag()
            # min_rotated_comparison = torch.min(rotated_looks_comparison)
            # max_rotated_comparison = torch.max(rotated_looks_comparison)
            # logging.info("Min rotated look comparison: {}".format(min_rotated_comparison))
            # logging.info("Max rotated look comparison: {}".format(max_rotated_comparison))
            # reversed_looks = rotate_vector(rotated_looks_from_angles, -rot_angles[:,0], -rot_angles[:,1])
            # original_looks_comparison = torch.mm(looks, reversed_looks.transpose(1,0)).diag()
            # min_orig_comparison = torch.min(original_looks_comparison)
            # max_orig_comparison = torch.max(original_looks_comparison)
            # logging.info("Min original look comparison: {}".format(min_orig_comparison))
            # logging.info("Max original look comparison: {}".format(max_orig_comparison))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
