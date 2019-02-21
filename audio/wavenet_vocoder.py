import os.path
import subprocess

from audio.base_vocoder import BaseVocoder

def WavenetVocoder(BaseVocoder):
    @staticmethod
    def modify_commandline_options(parser, isTrain):
        parser.add_argument('--wavenet_path', type=str, default='', help='path to wavenet folder')
        parser.add_argument('--wavenet_preset', type=str, default='', help='path to wavenet preset relative to wavenet_path')
        parser.add_argument('--wavenet_model', type=str, default='', help='path to wavenet model relative to wavenet_path')
        parser.add_argument('--absolute_paths', action='store_true', help='use the absolute paths for wavenet_preset and wavenet_model')
        return parser
        
    def initialize(self, opt):
        self.opt = opt
        self.wavenet_path = opt.wavenet_path
        if absolute_paths:
            self.wavenet_preset = opt.wavenet_preset
            self.wavenet_model = opt.wavenet_model
        else
            self.wavenet_preset = os.path.join(self.wavenet_path,opt.wavenet_preset)
            self.wavenet_model = os.path.join(self.wavenet_path,opt.wavenet_model)
        
    def synthesize(self, in_filename, out_filename, speaker_id):
        cmd_path = os.path.join(self.wavenet_path,'synthesis.py')
        cmd = [cmd_path,
               self.model_path,
               '.',
               '--conditional', in_filename,
               '--speaker-id', str(speaker_id),
               '--preset', self.wavenet_preset,]
        subprocess.run(cmd)
        os.rename(os.path.splitext(os.path.basename(checkpoint_path))[0]+'.wav',out_filename)
 
