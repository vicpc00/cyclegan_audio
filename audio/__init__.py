import importlib
from audio.base_vocoder import BaseVocoder


def find_vocoder_using_name(vocoder_name):
    """Import the module "audio/[vocoder_name]_vocoder.py".

    In the file, the class called VocoderNameVocoder() will
    be instantiated. It has to be a subclass of BaseVocoder,
    and it is case-insensitive.
    """
    vocoder_filename = "audio." + vocoder_name + "_vocoder"
    vocoderlib = importlib.import_module(vocoder_filename)

    vocoder = None
    target_vocoder_name = vocoder_name.replace('_', '') + 'vocoder'
    for name, cls in vocoderlib.__dict__.items():
        if name.lower() == target_vocoder_name.lower() \
           and issubclass(cls, BaseVocoder):
            vocoder = cls

    if vocoder is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseVocoder with class name that matches %s in lowercase." % (vocoder_filename, target_vocoder_name))

    return vocoder

def get_option_setter(vocoder_name):
    """Return the static method <modify_commandline_options> of the model class."""
    vocoder_class = find_vocoder_using_name(vocoder_name)
    return vocoder_class.modify_commandline_options

def create_vocoder(opt):

    vocoder = find_vocoder_using_name(opt.vocoder)
    instance = vocoder(opt)
    
    return instance

    
