import os.path
import subprocess
import numpy as np
import librosa
import soundfile as sf

from .base_vocoder import BaseVocoder

class GriffinLimVocoder(BaseVocoder):
    @staticmethod
    def modify_commandline_options(parser, isTrain):
        parser.add_argument('--sample_rate', type=int, default=16000, help='Sample rate of signals')
        parser.add_argument('--win_len', type=float, default=0.03, help='Window lenght in seconds')
        parser.add_argument('--hop_len', type=float, default=0.015, help='Window hop length in seconds')
        parser.add_argument('--nfft', type=int, default=None, help='Size of FFT. If not set defaults to window lenght in samples')
        parser.add_argument('--num_iter', type=int, default=500, help='Number of iterations')
        return parser
        
    def __init__(self, opt):
        BaseVocoder.__init__(self,opt)
        self.opt = opt
        self.sr = opt.sample_rate
        self.num_iter = opt.num_iter
        self.win_len = int(opt.win_len*self.sr)
        self.hop_len = int(opt.hop_len*self.sr)
        self.nfft = opt.nfft
        if self.nfft is None or self.nfft < self.win_len:
            self.nfft = self.win_len

    def analysis(self, in_filename):
        signal_dict = {}

        sig, sr = sf.read(in_filename)
        librosa.resample(sig,sr,self.sr)

        S = librosa.stft(sig, n_fft = self.nfft, win_length = self.win_len, hop_length = self.hop_len)
        S = np.abs(S).astype(np.float32)
        #lws_proc = lws.lws(self.win_len, self.hop_len, mode='speech', fftsize = self.nfft)
        #S = lws_proc.stft(s)
        #S = np.abs(S).T.astype(np.float32)
        print(S)
        S = self.amp_to_db_norm(S)

        signal_dict['tf_rep']  = S

        return signal_dict
    
    def synthesize(self, signal_dict, out_filename):
        #signal_dict = np.load(in_filename)
        mag_spec = self.db_to_amp_norm(signal_dict['tf_rep'])
        
        n_win = mag_spec.shape[1]
        sig_len = int((n_win-1)*self.hop_len)
        
        sig_rec = np.random.randn(sig_len)
        
        for i in range(self.num_iter):
            spec_rec = librosa.stft(sig_rec, n_fft = self.nfft, win_length = self.win_len, hop_length = self.hop_len)
            ang_spec = np.angle(spec_rec)
            spec_rec = mag_spec*np.exp(1.0j*ang_spec)
            sig_rec = librosa.istft(spec_rec, win_length = self.win_len, hop_length = self.hop_len)
            
        #librosa.output.write_wav(out_filename, sig_rec, self.sr)
        sf.write(out_filename, sig_rec, self.sr)
 
