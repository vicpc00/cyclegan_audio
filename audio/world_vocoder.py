import os.path
import subprocess
import numpy as np
import librosa
import pyworld as pw
import soundfile as sf

from audio.base_vocoder import BaseVocoder

class WorldVocoder(BaseVocoder):
    @staticmethod
    def modify_commandline_options(parser, isTrain):
        parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate of signals')
        #parser.add_argument('--win_len', type=float, default=0.03, help='Window lenght in seconds')
        parser.add_argument('--hop_len', type=float, default=0.015, help='Window hop length in seconds')
        parser.add_argument('--f0_floor', type=float, default=50.0, help='Lowest fundamenta frequency detected in hertz')
        parser.add_argument('--f0_ceil', type=float, default=600.0, help='Highest fundamenta frequency detected in hertz')
        return parser
        
    def __init__(self, opt):
        BaseVocoder.__init__(self,opt)
        self.opt = opt
        self.sr = opt.sample_rate
        self.hop_len = opt.hop_len
        self.f0_floor = opt.f0_floor
        self.f0_ceil = opt.f0_ceil

    def analysis(self, in_filename):
        signal_dict = {}

        sig, sr = sf.read(in_filename)
        librosa.resample(sig,sr,self.sr)

        _f0, t = pw.dio(sig, self.sr, f0_floor=self.f0_floor, f0_ceil=self.f0_ceil,
                    channels_in_octave=2,
                    frame_period=1000*self.hop_len,
                    speed=1)
        signal_dict['f0'] = pw.stonemask(sig, _f0, t, self.sr)
        signal_dict['C']  = pw.cheaptrick(sig, signal_dict['f0'], t, sr)
        signal_dict['ap'] = pw.d4c(sig, signal_dict['f0'], t, sr)

        return signal_dict
        
        
    def synthesize(self, signal_dict, out_filename):
        #signal_dict = np.load(in_filename)
        
        sig_rec = pw.synthesize(signal_dict['f0'], signal_dict['C'], signal_dict['ap'], self.sr, 1000*self.hop_len)
            
        #librosa.output.write_wav(out_filename, sig_rec, self.sr)
        sf.write(out_filename, sig_rec, self.sr)
 
