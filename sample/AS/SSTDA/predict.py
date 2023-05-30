import os

import torch
import torch.nn as nn
import numpy as np
from  eval import segment_bars

def predict(model, model_dir, results_dir, epoch, actions_dict, device, sample_rate, args, batch_gen_tst):

    # collect arguments
    verbose = args.verbose
    use_best_model = args.use_best_model
    epoch=epoch
    # multi-GPU
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    with torch.no_grad():
        model.to(device)
        if use_best_model == 'source':
            model.load_state_dict(torch.load(model_dir + "/acc_best_source.model"))
        elif use_best_model == 'target':
            model.load_state_dict(torch.load(model_dir + "/acc_best_target.model"))
        else:
            model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        batch_gen_tst.reset()
        import time

        time_start = time.time()
        while batch_gen_tst.has_next():
            batch_input, batch_target, mask, vids = batch_gen_tst.next_batchP(1,'source')
            vid = vids[0] + '.mp4'
            input_x = torch.tensor(batch_input, dtype=torch.float)
            input_x.unsqueeze(0)
            input_x = input_x.to(device)
            predictions, _, _, _, _, _, _, _, _, _, _, _, _, _  = model(input_x,input_x,torch.ones(input_x.size()), torch.ones(input_x.size()),[0, 0], False)
            _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
            predicted = predicted.squeeze()
            recognition = []
            print(predicted.shape)
            segment_bars(results_dir+'{}.png'.format(vid),batch_target.squeeze().tolist(),predicted.squeeze().tolist())
            for i in range(predicted.size(0)):
                recognition = np.concatenate((recognition,
                                              [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate))
            f_name = vid.split('/')[-1].split('.')[0]
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            f_ptr = open(results_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
        time_end = time.time()
def predictG(model, model_dir, results_dir, epoch, actions_dict, device, sample_rate, args, batch_gen_tst):

    # collect arguments
    verbose = args.verbose
    use_best_model = args.use_best_model
    epoch=epoch
    # multi-GPU
    if args.multi_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.eval()

    with torch.no_grad():
        model.to(device)
        if use_best_model == 'source':
            model.load_state_dict(torch.load(model_dir + "/acc_best_source.model"))
        elif use_best_model == 'target':
            model.load_state_dict(torch.load(model_dir + "/acc_best_target.model"))
        else:
            model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
        batch_gen_tst.reset()
        import time

        time_start = time.time()
        while batch_gen_tst.has_next():
            batch_input, batch_target, mask, vids,bf,bi = batch_gen_tst.next_goat_batchP(1,'source')
            vid = vids[0] + '.mp4'
            print(vid)
            input_x = torch.tensor(batch_input, dtype=torch.float)
            input_x.unsqueeze(0)
            input_x = input_x.to(device)
            predictions, _, _, _, _, _, _, _, _, _, _, _, _, _  = model(input_x,input_x,torch.ones(input_x.size()), torch.ones(input_x.size()),[0, 0], False,bf,bi,bf,bi)
            _, predicted = torch.max(predictions[:, -1, :, :].data, 1)
            predicted = predicted.squeeze()
            recognition = []
            segment_bars(results_dir+'{}.png'.format(vid),batch_target.tolist(),predicted.tolist())
            for i in range(predicted.size(0)):
                recognition = np.concatenate((recognition,
                                              [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]] * sample_rate))
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            f_name = vid.split('/')[-1].split('.')[0]
            f_ptr = open(results_dir + "/" + f_name, "w")
            f_ptr.write("### Frame level recognition: ###\n")
            f_ptr.write(' '.join(recognition))
            f_ptr.close()
        time_end = time.time()


if __name__ == '__main__':
    pass
