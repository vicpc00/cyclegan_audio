"""This module implements an abstract base class (ABC) 'BaseVocoder' for vocoders.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseVocoder(ABC):
    """This class is an abstract base class (ABC) for vocoders.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseVocoder.__init__(self, opt).
    -- <__len__>:                       return the size of vocoder.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add vocoder-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new vocoder-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def synthesize(self, in_filename, out_filename):
        """Synthesize the audio.
        
        Parameters:
            in_filename     -- File containing the spectogram
            out_filename    -- Where to save the result"""
        
    def amp_to_db_norm(spec):
        min_level_db = self.opt.min_level_db
        ref_level_db = self.opt.ref_level_db
        
        min_level = np.exp(np.log(10)*min_level_db/20)
        
        spec = 20*np.log(np.maximum(spec,min_level))/np.log(10) - ref_level_db
        spec = (spec - min_level_db)/-min_level_db
        spec = np.clip(spec,0,1)
        
        
    def db_to_amp_norm(spec):
        min_level_db = self.opt.min_level_db
        ref_level_db = self.opt.ref_level_db

        spec = np.clip(spec,0,1)

        spec = spec*(-min_level_db)+min_level_db+ref_level_db
        
        spec = np.exp(spec/20*np.log(10))
        return spec


