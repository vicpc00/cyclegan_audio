import os.path
import subprocess
import numpy as np
import librosa
import pyworld as pw
import soundfile as sf

from audio.base_vocoder import BaseVocoder

def GriffinLimVocoder(BaseVocoder):
    @staticmethod
    def modify_commandline_options(parser, isTrain):
        return parser
        
    def initialize(self, opt):
        self.opt = opt
        self.sr = opt.sample_rate
        self.win_len = opt.win_len
        self.hop_len = opt.hop_len
        self.nfft = opt.nfft
        if self.nfft = -1:
            self.nfft = self.win_len

    def analysis(self, in_filename):
        signal_dict = {}

        sig, sr = sf.read(in_filename)
        librosa.resample(sig,sr,self.sr)

        _f0, t = pw.dio(x, fs, f0_floor=50.0, f0_ceil=600.0,
                    channels_in_octave=2,
                    frame_period=1000*self.hop_len/self.sr,
                    speed=1)
        signal_dict['f0'] = pw.stonemask(sig, _f0, t, self.sr)
        signal_dict['C']  = pw.cheaptrick(sig, signal_dict['f0'], t, sr)
        signal_dict['ap'] = pw.d4c(sig, signal_dict['f0'], t, sr)

        return signal_dict
        
        
    def synthesize(self, in_filename, out_filename):
        signal_dict = np.load(in_filename)
        
        sig_rec = pw.synthesize(signal_dict['f0'], signal_dict['C'], signal_dict['ap'], self.sr, 1000*self.hop_len/self.sr)
            
        #librosa.output.write_wav(out_filename, sig_rec, self.sr)
        librosa.write(out_filename, sig_rec, self.sr)
 
