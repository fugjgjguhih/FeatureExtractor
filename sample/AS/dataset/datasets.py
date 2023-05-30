# Adapted from the code for paper 'What and How Well You Performed? A Multitask Learning Approach to Action Quality Assessment'.
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import pickle as pkl
#from opts import *
from scipy import stats



class VideoDataset(Dataset):

    def __init__(self,mode,files,args):
        super(VideoDataset, self).__init__()

        # train or test
        self.mode = mode

        # loading annotations, I3D features, CNN features, boxes annotations, formation features and bp features
        self.args = args
        self.keys = pkl.load(open(files, 'rb'))
        self.boxes_dict = pkl.load(open('bbox.pkl', 'rb'))
        print(f'len of {self.mode}:', len(self.keys))

        # parameters of videos
        self.data_path = args.data_path
        self.length = args.length
        self.img_size = args.img_size
        self.num_boxes = args.num_boxes
        self.out_size = args.out_size
        self.num_selected_frames = args.num_selected_frames
        self.goat = args.goat
        with open('vlen.pkl', 'rb') as f:
            self.vdict = pkl.load(f)
    def load_idx(self, frames_path):
        image_list = sorted((glob.glob(os.path.join(frames_path, '*.jpg'))))
        self.length = len(image_list)
        return image_list
    def load_idx2(self, key):
        length = self.length
        vlen = self.vdict[key[0]+'_'+str(key[1])+'.mp4']
        if vlen >= length:
            start_frame = 0
            end_frame = vlen-1
            frame_list = np.linspace(start_frame, end_frame, length).astype(np.int)
            image_frame_idx = [frame_list[i] - start_frame for i in range(length)]
            return image_frame_idx
        else:
            T = vlen
            img_idx_list = np.arange(T)
            img_idx_list = img_idx_list.repeat(2)
            idx_list = np.linspace(0, T * 2 - 1, length).astype(np.int)
            image_frame_idx = [img_idx_list[idx_list[i]] for i in range(length)]
            return image_frame_idx
    def load_boxes(self, key, frame_p, out_size):  # T,N,4
        key_bbox_list = [(key[0], str(key[1]), str(i).zfill(4)) for i in range(len(frame_p))]
        N = self.num_boxes
        H, W = out_size
        boxes = []
        for key_bbox in key_bbox_list:
            person_idx_list = []
            for i, item in enumerate(self.boxes_dict[key_bbox]['box_label']):
                if item == 'person':
                    person_idx_list.append(i)
            tmp_bbox = []
            tmp_x1, tmp_y1, tmp_x2, tmp_y2 = 0, 0, 0, 0
            for idx, person_idx in enumerate(person_idx_list):
                if idx < N:
                    box = self.boxes_dict[key_bbox]['boxes'][person_idx]
                    box[:2] -= box[2:] / 2
                    x, y, w, h = box.tolist()
                    x = x * W
                    y = y * H
                    w = w * W
                    h = h * H
                    tmp_x1, tmp_y1, tmp_x2, tmp_y2 = x, y, x + w, y + h
                    tmp_bbox.append(torch.tensor([x, y, x + w, y + h]).unsqueeze(0))  # 1,4 x1,y1,x2,y2
            if len(person_idx_list) < N:
                step = len(person_idx_list)
                while step < N:
                    tmp_bbox.append(torch.tensor([tmp_x1, tmp_y1, tmp_x2, tmp_y2]).unsqueeze(0))  # 1,4
                    step += 1
            boxes.append(torch.cat(tmp_bbox).unsqueeze(0))  # 1,N,4
        boxes_tensor = torch.cat(boxes)
        return boxes_tensor

    def random_select_frames(self, video, image_frame_idx):
        length = self.length
        num_selected_frames = self.num_selected_frames
        select_list_per_clip = [i for i in range(16)]
        selected_frames_list = []
        selected_frames_idx = []
        for i in range(length // 10):
            random_sample_list = random.sample(select_list_per_clip, num_selected_frames)
            selected_frames_list.extend([video[10 * i + j].unsqueeze(0) for j in random_sample_list])
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in random_sample_list])
        selected_frames = torch.cat(selected_frames_list, dim=0)  # 540*t,C,H,W; t=num_selected_frames
        return selected_frames, selected_frames_idx

    def select_middle_frames(self, video, image_frame_idx):
        length = self.length
        num_selected_frames = self.num_selected_frames
        selected_frames_list = []
        selected_frames_idx = []
        for i in range(length // 10):
            sample_list = [16 // (num_selected_frames + 1) * (j + 1) - 1 for j in range(num_selected_frames)]
            selected_frames_list.extend([video[10 * i + j].unsqueeze(0) for j in sample_list])
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in sample_list])
        selected_frames = torch.cat(selected_frames_list, dim=0)  # 540*t,C,H,W; t=num_selected_frames
        return selected_frames, selected_frames_idx

    def random_select_idx(self, image_frame_idx):
        length = self.length
        num_selected_frames = self.num_selected_frames
        select_list_per_clip = [i for i in range(16)]
        selected_frames_idx = []
        for i in range(length // 10):
            random_sample_list = random.sample(select_list_per_clip, num_selected_frames)
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in random_sample_list])
        return selected_frames_idx

    def select_middle_idx(self, image_frame_idx):
        length = self.length
        num_selected_frames = self.num_selected_frames
        selected_frames_idx = []
        for i in range(length // 10):
            sample_list = [16 // (num_selected_frames + 1) * (j + 1) - 1 for j in range(num_selected_frames)]
            selected_frames_idx.extend([image_frame_idx[10 * i + j] for j in sample_list])
        return selected_frames_idx

    def load_goat_data(self, data: dict, key: tuple):
        if self.args.use_goat:
            if self.args.use_cnn_features:
                frames_path = os.path.join(self.data_path, key[0]+'_'+str(key[1]))
                image_frame_idx = self.load_idx(frames_path)  # T,C,H,W
                data['boxes'] = self.load_boxes(key, image_frame_idx, self.out_size) # 540*t,N,4
                data['cnn_features'] = torch.from_numpy(np.load(os.path.join('./features_all_frames',key[0]+'_'+str(key[1])+'.npy')))
        return data
    def load_goat_data2(self, data: dict, key: tuple):
        if self.args.use_goat:
            if self.args.use_cnn_features:
                image_frame_idx = self.load_idx2(key)  # T,C,H,W
                if self.args.random_select_frames:
                    selected_frames_idx = self.random_select_idx(image_frame_idx)
                else:
                    selected_frames_idx = self.select_middle_idx(image_frame_idx)
                data['boxes'] = self.load_boxes2(key, selected_frames_idx, self.out_size)  # 540*t,N,4
                f = torch.from_numpy(np.load(os.path.join('/mnt/disk_1/xiangwei/features_all_frames', key[0] + '_' + str(key[1]) + '.npy')))
                fs =torch.stack(list(f[i,:,:] for i in selected_frames_idx))
                data['cnn_features'] = fs
        return data
    def load_boxes2(self, key, image_frame_idx, out_size):  # T,N,4
        key_bbox_list = [(key[0], str(key[1]), str(i).zfill(4)) for i in image_frame_idx]
        N = self.num_boxes
        T = self.length
        H, W = out_size
        boxes = []
        for key_bbox in key_bbox_list:
            person_idx_list = []
            for i, item in enumerate(self.boxes_dict[key_bbox]['box_label']):
                if item == 'person':
                    person_idx_list.append(i)
            tmp_bbox = []
            tmp_x1, tmp_y1, tmp_x2, tmp_y2 = 0, 0, 0, 0
            for idx, person_idx in enumerate(person_idx_list):
                if idx < N:
                    box = self.boxes_dict[key_bbox]['boxes'][person_idx]
                    box[:2] -= box[2:] / 2
                    x, y, w, h = box.tolist()
                    x = x * W
                    y = y * H
                    w = w * W
                    h = h * H
                    tmp_x1, tmp_y1, tmp_x2, tmp_y2 = x, y, x + w, y + h
                    tmp_bbox.append(torch.tensor([x, y, x + w, y + h]).unsqueeze(0))  # 1,4 x1,y1,x2,y2
            if len(person_idx_list) < N:
                step = len(person_idx_list)
                while step < N:
                    tmp_bbox.append(torch.tensor([tmp_x1, tmp_y1, tmp_x2, tmp_y2]).unsqueeze(0))  # 1,4
                    step += 1
            boxes.append(torch.cat(tmp_bbox).unsqueeze(0))  # 1,N,4
        boxes_tensor = torch.cat(boxes)
        return boxes_tensor
    def next_batch(self,bz,key):
        # build data
        data = {} # 540,1024
        if self.goat ==0:
            data = self.load_goat_data(data, key)
        else:
            data = self.load_goat_data2(data,key)
        return data
    def __len__(self):
        return len(self.keys)
