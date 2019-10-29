import time
import os
from options.train_options import TrainOptions
from data import create_dataloader
from models import create_model
from util.visualizer import Visualizer
import torch

if __name__ == '__main__':
    opt = TrainOptions().parse()
    dataset = create_dataloader(opt)
    dataset_size = len(dataset)
    print('#training specs = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            
            #Create datapoint with just time-freq representation
            data_spec = data.copy()
            data_spec['A'] = data_spec['A']['tf_rep']
            data_spec['B'] = data_spec['B']['tf_rep']
            #print(torch.max(data_spec['A']),torch.min(data_spec['A']),torch.max(data_spec['B']),torch.min(data_spec['B']))
            model.set_input(data_spec)
            model.optimize_parameters()

            """
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)
            """

            if total_steps % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                save_suffix = 'iter_%d' % total_steps if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

    convert_f0 = True
    if convert_f0:
        f0_stats = {'A':{'mean':0,'var':0,'N':0},'B':{'mean':0,'var':0,'N':0}}
        for i,data in enumerate(dataset):
            for spk in ['A','B']:
                f0_sig = data[spk]['f0'].squeeze()
                for j in range(f0_sig.shape[0]):
                    if f0_sig[j] == 0:
                        continue
                    f0_stats[spk]['N'] = f0_stats[spk]['N'] + 1
                    m_ant = f0_stats[spk]['mean']
                    f0_stats[spk]['mean'] += (f0_sig[j] - m_ant)/f0_stats[spk]['N'] #M_k = M_{k-1} + (x_k - M_{k-1})/k
                    f0_stats[spk]['var'] += (f0_sig[j] - m_ant)*(f0_sig[j] - f0_stats[spk]['mean']) #S_k = S_{k-1} + (x_k - M_{k-1})*(x_k - M_k)

        f0_stats['A']['var'] = f0_stats['A']['var']/f0_stats['A']['N']
        f0_stats['B']['var'] = f0_stats['B']['var']/f0_stats['B']['N']

        print(f0_stats)

        torch.save(f0_stats,os.path.join(opt.checkpoints_dir, opt.name, 'f0_stats'))
