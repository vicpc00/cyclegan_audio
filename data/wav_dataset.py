import os.path
from data.base_dataset import BaseDataset, get_transform
#from data.image_folder import make_dataset
from data.spec_folder import make_dataset
from audio import create_vocoder
from PIL import Image
import numpy as np
import random

def default_adjust(spec):
    return np.pad(spec,[(0,int(4*np.ceil(spec.shape[0]/4))-spec.shape[0]),(0,int(4*np.ceil(spec.shape[1]/4))-spec.shape[1])],'constant')
    

class WavDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')

        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        #self.transform = get_transform(opt)
        self.transform = default_adjust
        self.vocoder = create_vocoder(opt)

    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        
        A_signal = self.vocoder.analysis(A_path)
        B_signal = self.vocoder.analysis(B_path)

        A_signal['tf_rep'] = self.transform(A_signal['tf_rep'])
        B_signal['tf_rep'] = self.transform(B_signal['tf_rep'])
        A_signal['tf_rep'] = np.expand_dims(A_signal['tf_rep'],axis=0)
        B_signal['tf_rep'] = np.expand_dims(B_signal['tf_rep'],axis=0)

        return {'A': A_signal, 'B': B_signal,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'WavDataset'
