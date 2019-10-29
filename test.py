import os
from options.test_options import TestOptions
from data import create_dataloader
from models import create_model
from util.visualizer import save_signals
from util import html
from audio import create_vocoder
import torch
import copy

if __name__ == '__main__':
    opt = TestOptions().parse()
    # hard-code some parameters for test
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True    # no flip
    opt.display_id = -1   # no visdom display
    dataset = create_dataloader(opt)
    model = create_model(opt)
    model.setup(opt)
    vocoder = create_vocoder(opt)
    original_spk = {'real_A':'A','fake_B':'A','rec_A':'A','real_B':'B','fake_A':'B','rec_B':'B',}
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # pix2pix: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # CycleGAN: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    f0_stats = torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'f0_stats'))
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        
        #Create datapoint with just time-freq representation
        data_spec = data.copy()
        data_spec['A'] = data_spec['A']['tf_rep']
        data_spec['B'] = data_spec['B']['tf_rep']
        
        model.set_input(data_spec)
        model.test()
        signals = model.get_current_signals()
        sig_path = model.get_signal_paths()
        
        
        for sig_name in signals.keys():
            #tmp = data[original_spk[sig_name]].copy()
            tmp = copy.deepcopy(data[original_spk[sig_name]])
            tmp['tf_rep'] = signals[sig_name]
            signals[sig_name] = tmp

            if 'fake' in sig_name:
                tgt = sig_name[-1]
                src = 'A' if tgt == 'B' else 'B'
                idx = signals[sig_name]['f0'] > 0
                signals[sig_name]['f0'][idx] = (signals[sig_name]['f0'][idx] - f0_stats[src]['mean'])*f0_stats[tgt]['var']/f0_stats[src]['var'] + f0_stats[tgt]['mean']


        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, sig_path))
        #save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        save_signals(webpage, signals, sig_path, vocoder = vocoder)
    # save the website
    webpage.save()
