import numpy as np
import librosa
import lws

def griffin_lim(mag_spec, n_fft, win_len, hop_len, iterations):
    n_win = mag_spec.shape[1]
    #signal_len = int(n_win*hop_len + n_fft)
    signal_len = int((n_win-1)*hop_len)

    sig_rec = np.random.randn(signal_len)
    
    for i in range(iterations):
        spec_rec = librosa.stft(sig_rec, n_fft = n_fft, win_length = win_len, hop_length = hop_len)
        ang_spec = np.angle(spec_rec)
        spec_rec_new = mag_spec*np.exp(1.0j*ang_spec)
        
        sig_rec = librosa.istft(spec_rec_new, win_length = win_len, hop_length = hop_len)
    return sig_rec

def lws_phase(mag_spec, win_len, hop_len):
    lws_proc = lws.lws(win_len, hop_len,mode='speech')
    spec_rec = lws_proc.run_lws(mag_spec.astype('double'))
    sig_rec = lws_proc.istft(spec_rec)
    return sig_rec

def db_to_amp(spec):
    min_level_db = -100
    ref_level_db = 20

    spec = spec*(-min_level_db)+min_level_db+ref_level_db
    
    mag_spec = np.exp(spec/20*np.log(10))
    return mag_spec

def save_waveform_from_spec(spec,path):
    
    n_fft = 512
    sr = 16000
    t_win = 0.026
    t_hop = 0.013
    win_len = int(t_win*sr)
    hop_len = int(t_hop*sr)
    iterations = 300
    
    #mag_spec = librosa.db_to_amplitude(spec)
    #sig_rec = griffin_lim(mag_spec, n_fft, win_len, hop_len, iterations)
    mag_spec = db_to_amp(spec)
    sig_rec = lws_phase(mag_spec.T, 512, 256)
    librosa.output.write_wav(path,sig_rec,sr)
    
