
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
from Swin import SwinTransformer3D
from tqdm import tqdm
import torch
from torchvideotransforms import video_transforms, volume_transforms
import numpy as np
import glob
from PIL import Image
import random
import argparse
import  pickle

def load_video(args, frames_path):
    transforms = args.transforms
    image_list = sorted((glob.glob(os.path.join(frames_path, '*.jpg'))))
    seq = [Image.open(image_list[i]).convert('RGB') for i in range(len(image_list))]
    return transforms(seq)



# Basic parameters
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
args.img_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = '/mnt/disk_1/xiangwei/Bridge-Prompt-master/data/breakfast/frames'
path = '/mnt/disk_1/xiangwei/Video_result/Series2021Leg5_tech'
save_path = '/mnt/disk_1/xiangwei/swin_features'
args.transforms = video_transforms.Compose([video_transforms.Resize(args.img_size),
                                            volume_transforms.ClipToTensor(),
                                            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])

# Set pretrained model and position
# local_rank = torch.distributed.get_rank()
# torch.cuda.set_device(local_rank)
# device = torch.device("cuda", local_rank)
# torch.distributed.init_process_group(backend="nccl")
url = 'https://download.openmmlab.com/mmaction/v1.0/recognition/swin/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb/swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb_20220930-f8d74db7.pth'
# SwinTransformer3D(arch='l', pretrained=url)
# backbone = torch.nn.parallel.DistributedDataParallel(SwinTransformer3D(arch='l')) # device_ids will include all GPU devices by default
backbone = SwinTransformer3D(arch='l')
backbone.eval()
backbone.to(device)

# Inference
competition_list = os.listdir(path)
save_dir = '/mnt/disk_1/xiangwei/SWIN_Feature'
with open('vlen.pkl', 'rb') as f:
    vlen_dict = pickle.load(f)
c = 0
os.makedirs(save_dir)
print(os.environ["CUDA_VISIBLE_DEVICES"])
for v in tqdm(competition_list):
    ts = v.split('_')
    st = [s + '_' for s in ts]
    sts = [st[i] for i in range(len(st) - 1)]
    sss = '_'
    for s in sts:
        sss += s
    competition = sss.strip('_')
    team = ts[-1]
    with torch.no_grad():
        video_path = os.path.join(path, v)
        if os.path.exists(os.path.join(save_dir, competition + '_' + team + '.npy')):
            c = c + 1
            print(c)
            continue

        if len(os.listdir(video_path)) > 1:  # and (competition, int(team)) not in video_feamap_dict_key:
            video = load_video(args,video_path)
            video = video.unsqueeze(0)
            # 1,C,T,H,W
            print(video.shape)
            vname = competition + '_' + team + '.mp4'
            vlen = vlen_dict[vname]
            start_idx = list(range(16, vlen - 16, 1))
            v_head = [video[:, :, i: i + 32] for i in range(16)]
            v_body = [video[:, :, i - 16: i + 16] for i in start_idx]
            v_back = [video[:, :, vlen + i - 48: vlen + i - 16] for i in range(16)]
            video_pack = torch.cat(v_head + v_body + v_back)
            image_inputs = torch.split(video_pack,16)
            video_feature = []
            for inp in image_inputs:
                inp = inp.cuda()
                feature = backbone(inp).mean(2).mean(2).mean(2)
                print(feature.shape)
                video_feature.append(feature.to('cpu'))
            video_feature = torch.cat(video_feature).view(vlen, 1536).numpy()
            print(video_feature.shape)
            np.save(os.path.join(save_dir, competition + '_' + team + '.npy'), video_feature)
