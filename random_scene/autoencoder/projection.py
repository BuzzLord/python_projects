
from os.path import join
import matplotlib.pyplot as plt
import cv2
from gl_math import *


class Projection:
    def __init__(self, xrot=0.0, yrot=0.0, fov=90.0, aspect=1.0):
        self.basis = rotx(xrot) * roty(yrot) * np.identity(4)
        self.fov = fov
        self.aspect = aspect

    def generate_map(self, output_dim=(256,256), input_dim=None):
        if input_dim is None:
            input_dim = output_dim

        ray = np.array([0.0,0.0,-1.0,0.0], dtype=np.float32)
        rot_left = roty((self.fov/self.aspect)/2.0)
        rot_up = rotx(self.fov/2.0)
        top_left = transform(self.basis, transform(rot_left, transform(rot_up, ray)))
        top_right = transform(self.basis, transform(rot_left.T, transform(rot_up, ray)))
        bot_left = transform(self.basis, transform(rot_left, transform(rot_up.T, ray)))
        bot_right = transform(self.basis, transform(rot_left.T, transform(rot_up.T, ray)))
        tl_matrix = np.asmatrix(np.linspace(1.0, 0.0, output_dim[1])).T * np.asmatrix(np.linspace(1.0, 0.0, output_dim[0]))
        tr_matrix = np.asmatrix(np.linspace(0.0, 1.0, output_dim[1])).T * np.asmatrix(np.linspace(1.0, 0.0, output_dim[0]))
        bl_matrix = np.asmatrix(np.linspace(1.0, 0.0, output_dim[1])).T * np.asmatrix(np.linspace(0.0, 1.0, output_dim[0]))
        br_matrix = np.asmatrix(np.linspace(0.0, 1.0, output_dim[1])).T * np.asmatrix(np.linspace(0.0, 1.0, output_dim[0]))

        tl_matrix = np.expand_dims(tl_matrix, -1)
        tl_matrix = (tl_matrix * np.ones((1, 1, 3))) * top_left[0:3]

        tr_matrix = np.expand_dims(tr_matrix, -1)
        tr_matrix = (tr_matrix * np.ones((1, 1, 3))) * top_right[0:3]

        bl_matrix = np.expand_dims(bl_matrix, -1)
        bl_matrix = (bl_matrix * np.ones((1, 1, 3))) * bot_left[0:3]

        br_matrix = np.expand_dims(br_matrix, -1)
        br_matrix = (br_matrix * np.ones((1, 1, 3))) * bot_right[0:3]

        combined = tl_matrix + tr_matrix + bl_matrix + br_matrix
        comb_len = np.power(np.sum(np.power(combined, 2), axis=-1), 0.5)
        normed = combined / np.tile(comb_len[..., None], 3)

        theta = input_dim[0] * (1.0 - (np.arccos(normed[:,:,0]) / np.pi))
        # tmin, tmax = np.min(theta), np.max(theta)

        phi = input_dim[1] * (np.arctan2(-normed[:,:,2], normed[:,:,1]) / np.pi)
        # pmin, pmax = np.min(phi), np.max(phi)

        return phi.astype(np.float32), theta.astype(np.float32)

    def main(self, filename, map):
        org = cv2.imread(filename)[..., ::-1]
        interpolation = cv2.INTER_AREA
        remap = cv2.remap(org, map[0], map[1], interpolation=interpolation, borderValue=0, borderMode=cv2.BORDER_CONSTANT)

        combined = np.concatenate((org,remap), axis=1)
        plt.imshow(combined)
        plt.show()


if __name__ == "__main__":
    proj = Projection(xrot=0.0, yrot=90.0, aspect=1.0, fov=60)
    filename = join(join("..", "screens2_512"), "ss2_001760r_1.00_0.00_0.00.png")
    phi, theta = proj.generate_map(output_dim=(512,512), input_dim=(512,512))
    proj.main(filename, (phi, theta))
