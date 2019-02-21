import os.path
import subprocess
import numpy as np
import librosa

from audio.base_vocoder import BaseVocoder

def GriffinLimVocoder(BaseVocoder):
    @staticmethod
    def modify_commandline_options(parser, isTrain):
        parser.add_argument('--num_iter', type=int, default=500, help='Number of iterations')
        return parser
        
    def initialize(self, opt):
        self.opt = opt
        self.sr = opt.sample_rate
        self.num_iter = opt.num_iter
        self.win_len = opt.win_len
        self.hop_len = opt.hop_len
        self.nfft = opt.nfft
        if self.nfft = -1:
            self.nfft = self.win_len
        
    def synthesize(self, in_filename, out_filename):
        mag_spec = np.load(in_filename)
        mag_spec = self.db_to_amp_norm(mag_spec)
        
        n_win = mag_spec.shape[1]
        sig_len = int((n_win-1)*hop_len)
        
        sig_rec = np.random.randn(sig_len)
        
        for i in range(self.num_iter):
            spec_rec = librosa.stft(sig_rec, n_fft = self.nfft, win_length = self.win_len, hop_length = self.hop_len)
            ang_spec = np.angel(spec_rec)
            spec_rec = mag_spec*np.exp(1.0j*ang_spec)
            sig_rec = librosa.istft(spec_rec, win_length = self.win_len, hop_length = self.hop_len)
            
        librosa.output.write_wav(out_filename, sig_rec, self.sr)
 
