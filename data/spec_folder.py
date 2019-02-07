###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

import numpy as np
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

EXTENSIONS = ['.npy']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in EXTENSIONS)


def make_dataset(dir):
    specs = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                specs.append(path)

    return specs


def default_loader(path):
    return np.load(path)

def default_adjust(spec):
    return np.pad(spec,[(0,4-spec.shape[0]%4),(0,4-spec.shape[1]%4)],'constant')


class SpecFolder(data.Dataset):

    def __init__(self, root, transform=default_adjust, return_paths=False,
                 loader=default_loader):
        specs = make_dataset(root)
        if len(specs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.specs = specs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.specs[index]
        spec = self.loader(path)
        if self.transform is not None:
            spec = self.transform(spec)
        if self.return_paths:
            return spec, path
        else:
            return spec

    def __len__(self):
        return len(self.specs)
