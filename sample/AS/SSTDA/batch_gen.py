'''
    Adapted from https://github.com/yabufarha/ms-tcn
'''

import torch
import numpy as np
import random
import pickle
from  datasets import  VideoDataset
class BatchGenerator(object):
    def __init__(self, num_classes, gt_path, features_path, sample_rate, args):
        self.index = 0
        self.num_classes = num_classes
        self.gt_path = gt_path
        self.sample_rate = sample_rate
        self.features_path = features_path
        self.args =args
    def reset(self):
        self.index = 0
        #self.my_shuffle()

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        with open(self.gt_path, 'rb') as f:
            gts = pickle.load(f)
        self.list_of_examples = []
        self.args=args
        # for vid in ls:
        #     if vid in gts:
        #         self.list_of_examples.append(vid)
        # # print(gts)
        # self.gts = [gts[vid][-1] for vid in self.list_of_examples]
        self.gts = [gts[vid] for vid in gts.keys()]
        # lss=[]
        # for vid in self.list_of_examples:
        #     ss = vid[-2]+'_'+str(vid[-1])
        #     lss.append(ss)
        self.list_of_examples = list(gts.keys())
        self.features = [self.features_path + vid for vid in gts.keys()]
        self.my_shuffle()
        self.tdataset = VideoDataset(mode="train", args=self.args, files=vid_list_file)
    def my_shuffle(self):
        # shuffle list_of_examples, gts, features with the same order
        randnum = random.randint(0, 100)
        random.seed(randnum)
        random.shuffle(self.list_of_examples)
        random.seed(randnum)
        random.shuffle(self.gts)
        random.seed(randnum)
        random.shuffle(self.features)
    def merge(self, bg, suffix):
        '''
        merge two batch generator. I.E
        BatchGenerator a;
        BatchGenerator b;
        a.merge(b, suffix='@1')
        :param bg:
        :param suffix: identify the video
        :return:
        '''

        self.list_of_examples += [vid + suffix for vid in bg.list_of_examples]
        self.gts += bg.gts
        self.features += bg.features

        print('Merge! Dataset length:{}'.format(len(self.list_of_examples)))

    def next_goat_batch(self, batch_size, flag,if_warp=False):  # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]
        self.index += batch_size
        if flag == 'target' and self.index == len(self.list_of_examples):
            self.reset()

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features = np.load(batch_features[idx])
            target = np.array(batch_gts[idx])
            feature = features[:, ::self.sample_rate].T
            target = target[::self.sample_rate]
            batch_input.append(feature)
            batch_target.append(target)
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences),
                                         dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(batch_input[i]).unsqueeze(0),
                                                              torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i,
                                                                        :np.shape(batch_target[i])[
                                                                            0]] = warped_input.squeeze(
                    0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
            batch_target_tensor = torch.autograd.Variable(batch_target_tensor) - 1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if batch_size > 1:
                boxes_features = []
                boxes_in = []
                for vids in batch:
                    ts = vids[0].split('_')
                    st = [s + '_' for s in ts]
                    sts = [st[i] for i in range(len(st) - 1)]
                    sss = '_'
                    for s in sts:
                        sss += s
                    sss = sss.strip('_')
                    data = self.tdataset.next_batch(batch_size, (sss, int(ts[-1])))
                    boxes_features.append(data['cnn_features'])
                    boxes_in.append(data['boxes'])  # B,T,N,4
                boxes_in_1 = torch.stack(boxes_in).to(device)
                boxes_features_1 = torch.stack(boxes_features).to(device)
                batch_input_tensor = batch_input_tensor.permute(0, 2, 1)
            else:
                ts = batch[0].split('_')
                st = [s + '_' for s in ts]
                sts = [st[i] for i in range(len(st) - 1)]
                sss = '_'
                for s in sts:
                    sss += s
                sss = sss.strip('_')
                data = self.tdataset.next_batch(batch_size, (sss, int(ts[-1])))
                boxes_features_1 = data['cnn_features'].to(device).unsqueeze(0)
                boxes_in_1 = data['boxes'].to(device).unsqueeze(0)  # B,T,N,4
            return batch_input_tensor, batch_target_tensor, mask,boxes_features_1,boxes_in_1
    def next_batch(self, batch_size,flag,if_warp=False):  # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]
        self.index += batch_size
        if flag == 'target' and self.index == len(self.list_of_examples):
            self.reset()

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features = np.load(batch_features[idx])
            target = np.array(batch_gts[idx])
            feature = features[:, ::self.sample_rate].T
            target = target[::self.sample_rate]
            batch_input.append(feature)
            batch_target.append(target)
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences),
                                         dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(batch_input[i]).unsqueeze(0),
                                                              torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i,
                                                                        :np.shape(batch_target[i])[
                                                                            0]] = warped_input.squeeze(
                    0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        return batch_input_tensor, batch_target_tensor, mask
    def next_goat_batchP(self, batch_size, flag,
                   if_warp=False):  # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]
        self.index += batch_size
        if flag == 'target' and self.index == len(self.list_of_examples):
            self.reset()

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features = np.load(batch_features[idx])
            target = np.array(batch_gts[idx])
            feature = features[:, ::self.sample_rate].T
            target = target[::self.sample_rate]
            batch_input.append(feature)
            batch_target.append(target)
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences),
                                         dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(batch_input[i]).unsqueeze(0),
                                                              torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i,
                                                                        :np.shape(batch_target[i])[
                                                                            0]] = warped_input.squeeze(
                    0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
            batch_target_tensor = torch.autograd.Variable(batch_target_tensor) - 1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if batch_size > 1:
                boxes_features = []
                boxes_in = []
                for vids in batch:
                    ts = vids[0].split('_')
                    st = [s + '_' for s in ts]
                    sts = [st[i] for i in range(len(st) - 1)]
                    sss = '_'
                    for s in sts:
                        sss += s
                    sss = sss.strip('_')
                    data = self.tdataset.next_batch(batch_size, (sss, int(ts[-1])))
                    boxes_features.append(data['cnn_features'])
                    boxes_in.append(data['boxes'])  # B,T,N,4
                boxes_in_1 = torch.stack(boxes_in).to(device)
                boxes_features_1 = torch.stack(boxes_features).to(device)
                batch_input_tensor = batch_input_tensor.permute(0, 2, 1)
            else:
                ts = batch[0].split('_')
                st = [s + '_' for s in ts]
                sts = [st[i] for i in range(len(st) - 1)]
                sss = '_'
                for s in sts:
                    sss += s
                sss = sss.strip('_')
                data = self.tdataset.next_batch(batch_size, (sss, int(ts[-1])))
                boxes_features_1 = data['cnn_features'].to(device).unsqueeze(0)
                boxes_in_1 = data['boxes'].to(device).unsqueeze(0)  # B,T,N,4
        return batch_input_tensor, batch_target_tensor, mask,batch,boxes_features_1,boxes_in_1
    def next_batchP(self, batch_size, flag,
                   if_warp=False):  # if_warp=True is a strong data augmentation. See grid_sampler.py for details.
        batch = self.list_of_examples[self.index:self.index + batch_size]
        batch_gts = self.gts[self.index:self.index + batch_size]
        batch_features = self.features[self.index:self.index + batch_size]
        self.index += batch_size
        if flag == 'target' and self.index == len(self.list_of_examples):
            self.reset()

        batch_input = []
        batch_target = []
        for idx, vid in enumerate(batch):
            features = np.load(batch_features[idx])
            target = np.array(batch_gts[idx])
            feature = features[:, ::self.sample_rate].T
            target = target[::self.sample_rate]
            batch_input.append(feature)
            batch_target.append(target)
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences),
                                         dtype=torch.float)  # bs, C_in, L_in
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long) * (-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            if if_warp:
                warped_input, warped_target = self.warp_video(torch.from_numpy(batch_input[i]).unsqueeze(0),
                                                              torch.from_numpy(batch_target[i]).unsqueeze(0))
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]], batch_target_tensor[i,
                                                                        :np.shape(batch_target[i])[
                                                                            0]] = warped_input.squeeze(
                    0), warped_target.squeeze(0)
            else:
                batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
                batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
            batch_target_tensor = torch.autograd.Variable(batch_target_tensor) - 1
        return batch_input_tensor, batch_target_tensor, mask,batch

if __name__ == '__main__':
    pass