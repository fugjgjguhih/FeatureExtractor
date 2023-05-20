import os
import Gconfig
args = Gconfig.get_parser()
mymodel = args.model
num_epochs = args.epochs
gpu = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
if args.action =='predict':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import torch
from SSTDA.model import MultiStageModel,GOAT
from SSTDA.train import Trainer
from SSTDA.predict import *
from SSTDA.batch_gen import *
import argparse
import random
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# seed = args.seed

seed =1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


# check whether place_adv & place_sem are valid
if len(args.place_adv) != args.num_stages:
    raise ValueError('len(place_dis) should be equal to num_stages')
if len(args.place_sem) != args.num_stages:
    raise ValueError('len(place_sem) should be equal to num_stages')
if len(args.place_ent) != args.num_stages:
    raise ValueError('len(place_ent) should be equal to num_stages')
if len(args.place_dis) != args.num_stages:
    raise ValueError('len(place_dis) should be equal to num_stages')
if len(args.place_ens) != args.num_stages:
    raise ValueError('len(place_ens) should be equal to num_stages')
if len(args.place_ss) != args.num_stages:
    raise ValueError('len(place_ss) should be equal to num_stages')

if args.use_target == 'none':
    args.DA_adv = 'none'
    args.DA_sem = 'none'
    args.DA_ent = 'none'
    args.DA_dis = 'none'
    args.DA_ens = 'none'
    args.SS_video = 'none'  # focus on cross-domain setting

# use the full temporal resolution @ 15fps
sample_rate = 1
# sample input features @ 15fps instead of 30 fps
# for 50salads, and up-sample the output to 30 fps
if args.dataset == "50salads":
    sample_rate = 2

# ====== Load files ====== #
vid_list_file = "train_split.pkl"
vid_list_file_tst = "test_split.pkl"
feature_dim = 1024
if args.feature=='BP':
    feature_dim=768
    features_path = "./data/" + args.dataset + "/features/"
elif args.feature=='swin':
    feature_dim=1536
    features_path = '/mnt/disk_1/xiangwei/SWIN_Feature/'
else:
    features_path = '/mnt/disk_1/xiangwei/I3D_Feat/'
vid_list_file_target=vid_list_file_tst
gt_path = "anno_dict.pkl"

# mapping_file = "./data/"+args.dataset+"/mapping.txt"

model_dir = "./SSTDA{}models/breakfast/split_".format(args.feature) + args.split

results_dir = "./SSTDA{}results/breakfast/split_".format(args.feature) + args.split

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

actions_dict = {
    'Required element 1': 0,
    'Required element 2': 1,
    'Required element 3': 2,
    'Required element 4': 3,
    'Required element 5': 4,
    'Acrobatic movements': 5,
    'Cadence action': 6,
    'Free': 7,
    'Upper': 8,
    'Lower': 9,
    'Float': 10,
    'None': 11
}
print(os.environ["CUDA_VISIBLE_DEVICES"])

num_classes = len(actions_dict)


# initialize model & trainer
if  args.goat == 2:
    print(args.goat)
    model = MultiStageModel(args, num_classes,feature_dim)
else:
    model = GOAT(args, num_classes,feature_dim)
trainer = Trainer(num_classes)

# ====== Main Program ====== #
start_time = time.time()
if not args.goat == 2:
    model_dir = "./SSTDA{}goatmodels/breakfast/split_".format(args.feature) + args.split
    results_dir = "./SSTDA{}goatresults/breakfast/split_".format(args.feature) + args.split
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        os.makedirs(results_dir)
    if args.action == "train":
        batch_gen_source = BatchGenerator(num_classes, gt_path, features_path, sample_rate,args)
        batch_gen_target = BatchGenerator(num_classes, gt_path, features_path, sample_rate,args)
        batch_gen_source.read_data(vid_list_file)  # read & shuffle the source training list
        batch_gen_target.read_data(vid_list_file_tst)
        # read & shuffle the target training list
        trainer.trainG(model, model_dir, results_dir, batch_gen_source, batch_gen_target, device, args)

    if args.action == "predict":
        batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate,args)
        batch_gen_tst.read_data(vid_list_file_tst)
        predictG(model, model_dir, results_dir, num_epochs, actions_dict,
                device, sample_rate, args, batch_gen_tst)
else:
    if args.action == "train":
        batch_gen_source = BatchGenerator(num_classes, gt_path, features_path, sample_rate,args)
        batch_gen_target = BatchGenerator(num_classes, gt_path, features_path, sample_rate,args)
        batch_gen_source.read_data(vid_list_file)  # read & shuffle the source training list
        batch_gen_target.read_data(vid_list_file_target)  # read & shuffle the target training list
        trainer.train(model, model_dir, results_dir, batch_gen_source, batch_gen_target, device, args)

    if args.action == "predict":
        batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate,args)
        batch_gen_tst.read_data(vid_list_file_tst)
        predict(model, model_dir, results_dir,  num_epochs, actions_dict,
                device, sample_rate, args,batch_gen_tst)

end_time = time.time()

if args.verbose:
    print('')
    print('total running time:', end_time - start_time)
