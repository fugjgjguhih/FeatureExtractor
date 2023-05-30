
import os
os.environ["CUDA_VISIBLE_DEVICES"]="7"
from model.Swin import SwinTransformer3D
from tqdm import tqdm
import torch
from torchvideotransforms import video_transforms, volume_transforms
import numpy as np
import glob
from PIL import Image
import torch.nn as nn
import argparse
import  pickle
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--fpath', default='/mnt/disk_1/xiangwei/LL/frame')
parser.add_argument('--spath', default='/mnt/disk_1/xiangwei/LL/SWIN_Feature2')
parser.add_argument('--scale', type=int default=8)
args = parser.parse_args()
checkpoint='/mnt/disk_1/xiangwei/TransRAC-main/pretrained/swin_tiny_patch244_window877_kinetics400_1k.pth'
videop = '/mnt/disk_1/xiangwei/LL/video'
Swin_arch='t'
scale=args.scale
feature_dim=768
if Swin_arch=='l':
    feature_dim=1536

folders=['train','valid','test']
def load_video(args, frames_path):
    transforms = args.transforms
    image_list = sorted((glob.glob(os.path.join(frames_path, '*.jpg'))))
    seq = [Image.open(image_list[i]).convert('RGB') for i in range(len(image_list))]
    return transforms(seq)

save_dir = args.spath
path = args.fpath
# Basic parameters
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int)
args = parser.parse_args()
args.img_size = (224, 224)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


args.transforms = video_transforms.Compose([video_transforms.Resize(args.img_size),
                                            volume_transforms.ClipToTensor(),
                                            video_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        #self.backbone = SwinTransformer3D(arch='t', drop_path_rate=0.4)
        self.backbone = SwinTransformer3D(arch=Swin_arch, drop_path_rate=0.4)

    def forward(self, x):
        return self.backbone(x)


backbone = Model()
# edit to load pretrained 
state_dict = torch.load(checkpoint)
state_dict = state_dict['state_dict']
backbone.load_state_dict(state_dict, strict=False)

assert torch.equal(backbone.backbone.patch_embed.proj.weight.data, state_dict['backbone.patch_embed.proj.weight'])
backbone.eval()
backbone.cuda()


# Inference
with open('vlen.pkl', 'rb') as f:
    vlen_dict = pickle.load(f)
c = 0
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
print(os.environ["CUDA_VISIBLE_DEVICES"])
for fold in folders:
    video_list = os.listdir(os.path.join(path,fold))
    # feature_path = os.path.join(save_dir,fold)
    feature_path = save_dir
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    for vname in tqdm(video_list):
        video_path=os.path.join(path,fold,vname)
        print(os.path.join(feature_path, vname+'.npy'))
        with torch.no_grad():
            if len(os.listdir(video_path)) > 1: 
                if not os.path.exists(os.path.join(feature_path, vname+'.npy')):            
                    video = load_video(args,video_path)
                    video = video.unsqueeze(0)
                    # 1,C,T,H,W
                    vlen = vlen_dict[os.path.join(videop,fold,vname+'.mp4')]
                    start_idx = list(range(scale, vlen - scale, 1))
                    v_head = [video[:, :, i: i + scale*2] for i in range(scale)]
                    v_body = [video[:, :, i - scale: i + scale] for i in start_idx]
                    v_back = [video[:, :, vlen + i - scale*3: vlen + i - scale] for i in range(scale)]
                    video_pack = torch.cat(v_head + v_body + v_back)
                    image_inputs = torch.split(video_pack,scale)
                    video_feature = []
                    for inp in image_inputs:
                        inp = inp.cuda()
                        feature = backbone(inp).mean(2).mean(2).mean(2)
                        video_feature.append(feature)
                    video_feature = torch.cat(video_feature).view(vlen,feature_dim).to('cpu').numpy()
                    print(video_feature.shape)

                    np.save(os.path.join(feature_path, vname+'.npy'), video_feature)
                    gc.collect()
