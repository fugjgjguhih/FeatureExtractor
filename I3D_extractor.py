import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
import torch
from PIL import Image
import glob
import numpy as np
from torchvideotransforms import video_transforms, volume_transforms
from models.i3d import I3D
import pickle
from tqdm import tqdm

length = 5406
transforms = video_transforms.Compose([
    video_transforms.RandomHorizontalFlip(),
    video_transforms.Resize((112, 112)),
    volume_transforms.ClipToTensor(),
    video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_trans = video_transforms.Compose([
    video_transforms.Resize((112, 112)),
    volume_transforms.ClipToTensor(),
    video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_video(frames_path):
    global length
    image_list = sorted((glob.glob(os.path.join(frames_path, '*.jpg'))))
    if len(image_list) >= length:
        start_frame = int(image_list[0].split("/")[-1][:-4])
        end_frame = int(image_list[-1].split("/")[-1][:-4])
        frame_list = np.linspace(start_frame, end_frame, length).astype(np.int)
        image_frame_idx = [frame_list[i] - start_frame for i in range(length)]
        video = [Image.open(image_list[image_frame_idx[i]]) for i in range(length)]
        return transforms(video)
    else:
        video = [Image.open(image_list[i]) for i in range(len(image_list))]
        video_1 = transforms(video)
        C, T, H, W = video_1.size()
        video_1 = video_1.unsqueeze(-1).expand(C, T, H, W, 2)
        video_1 = video_1.reshape(C, -1, H, W)
        C1, T1, H1, W1 = video_1.size()

        select_space = np.linspace(0, T1 - 1, length).astype(np.int)
        select_frame_list = [select_space[i] for i in range(length)]

        video_1 = torch.cat([video_1[:, ii, :, :].unsqueeze(1) for ii in select_frame_list], 1)
        return video_1


backbone = I3D(num_classes=400, modality='rgb', dropout_prob=0.5)
I3D_ckpt_path = 'models/model_rgb.pth'
backbone.load_state_dict(torch.load(I3D_ckpt_path))
use_gpu = torch.cuda.is_available()
if use_gpu:
    backbone = backbone.cuda()

device = 'cpu'
with open('video_feature_dict.pkl', 'rb') as f:
    video_feature_dict = pickle.load(f)
with open('video_feamap_dict.pkl', 'rb') as f:
    video_feamap_dict = pickle.load(f)
video_feature_dict_key = video_feature_dict.keys()
video_feamap_dict_key = video_feamap_dict.keys()
path = '/mnt/petrelfs/daiwenxun/AS-AQA/Video_result'
competition_list = os.listdir(path)
for competition in competition_list:
    team_path = os.path.join(path, competition)
    team_list = os.listdir(team_path)
    for team in tqdm(team_list):
        with torch.no_grad():
            video_path = os.path.join(team_path, team)
            if len(os.listdir(video_path)) > 1000 and (competition, int(team)) not in video_feamap_dict_key:
                video = load_video(video_path)
                print('loading ckpt done')

                # 1:5760,9,960,360
                # 2,5406,540,16,6
                video = video.unsqueeze(0)
                # 1,C,T,H,W
                video = video.cuda()
                start_idx = list(range(0, 5400, 10))
                video_pack = torch.cat([video[:, :, i: i + 16] for i in start_idx])
                video_feamap, video_feature = backbone(video_pack)
                # 540,1024,1,1,1
                video_feature_dict[(competition, int(team))] = video_feature.to(device)
                video_feamap_dict[(competition, int(team))] = video_feamap.to(device)
            else:
                print(video_path + ' length < 1000')

with open('video_feature_dict.pkl', 'wb') as f:
    pickle.dump(video_feature_dict, f)
with open('video_feamap_dict.pkl', 'wb') as f:
    pickle.dump(video_feamap_dict, f)
