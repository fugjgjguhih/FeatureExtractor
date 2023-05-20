import torch
 
from model import *
from batch_gen import BatchGenerator
from eval import func_eval

import os
import argparse
import numpy as np
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 19980125 # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 
parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="breakfast")
parser.add_argument('--split', default='0')
parser.add_argument('--model_dir', default='models')
parser.add_argument('--result_dir', default='results')
parser.add_argument('--epochs',default=90,type=int)

args = parser.parse_args()
 
num_epochs = args.epochs

lr = 0.0005
num_layers = 10
num_f_maps = 64
features_dim = 768
bz = 1

channel_mask_rate = 0.3


# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# To prevent over-fitting for GTEA. Early stopping & large dropout rate
if args.dataset == "gtea":
    channel_mask_rate = 0.5
    
if args.dataset == 'breakfast':
    lr = 0.0001


vid_list_file = "train_split.pkl"
vid_list_file_tst = "test_split.pkl"
features_path = "./data/"+args.dataset+"/features/"
gt_path = "anno_dict.pkl"
# mapping_file = "./data/"+args.dataset+"/mapping.txt"
 
model_dir = "./{}/".format(args.model_dir)+args.dataset+"/split_"+args.split

results_dir = "./{}/".format(args.result_dir)+args.dataset+"/split_"+args.split
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

actions_dict={
'Required element 1':0,
'Required element 2':1,
'Required element 3':2,
'Required element 4':3,
'Required element 5':4,
'Acrobatic movements':5,
'Cadence action':6,
'Free':7,
'Upper':8,
'Lower':9,
'Float':10,
'None':11
}
num_classes = 12
trainer = Trainer(num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate,'ASFormer')
if args.action == "train":
    batch_gen = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)

    trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

if args.action == "predict":
    batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
    batch_gen_tst.read_data(vid_list_file_tst)
    trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict, sample_rate)

