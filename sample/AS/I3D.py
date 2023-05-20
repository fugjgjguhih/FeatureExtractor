import os
import Gconfig
args = Gconfig.get_parser()
mymodel = args.model
# num_epochs =70
# if args.goat == 0:
#     num_epochs = 65
num_epochs = args.epochs
gpu = args.gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
# if args.action == 'predict' and args.goat == 0:
#      os.environ["CUDA_VISIBLE_DEVICES"] = '7'
from model import *
import model
import MSTCN2
from batch_gen import *
import random
from  eval import  func_eval
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if mymodel == 'ASFormer':
    seed = 19980125
else:
    seed = 1538574472
# seed =args.seed # my birthday, :)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
 

lr = 0.0005
num_layers = 10
num_f_maps = 64
features_dim = 1024
bz = args.batch_size

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
if args.feature == 'BP':
    features_dim=768
    features_path = "./data/" + args.dataset + "/features/"
elif args.feature =='swin':
    features_dim=1536
    features_path = './SWIN_Feature/'
else:
    features_path = './I3D_Feat/'
gt_path = "anno_dict.pkl"
 
# mapping_file = "./data/"+args.dataset+"/mapping.txt"
 
model_dir = "./{}{}{}/".format(args.model,args.model_dir, args.feature)+args.dataset+"/split_"+args.split

results_dir = "./{}{}{}/".format(args.model,args.result_dir, args.feature)+args.dataset+"/split_"+args.split
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
if mymodel == 'BCN':
    dd=1
else:
    if mymodel == 'ASFormer':
        model_dir = "./ASFormer{}{}/".format(args.model_dir,args.feature) + args.dataset + "/split_" + args.split

        results_dir = "./ASFormer{}{}/".format(args.result_dir, args.feature) + args.dataset + "/split_" + args.split+'/'
        bz=1
        trainer = model.Trainer(args,num_layers, 2, 2, num_f_maps, features_dim, num_classes, channel_mask_rate,mymodel)
        if args.goat==2:
            if args.action == "train":
                batch_gen = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen.read_data(args,vid_list_file)

                batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen_tst.read_data(args,vid_list_file_tst)

                trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

            if args.action == "predict":
                batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen_tst.read_data(args,vid_list_file_tst)
                file_list = "test_split.pkl"
                num_epochs = 40
                trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 60
                trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 70
                trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 80
                trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 90
                trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)
                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
        else:
            if args.action == "train":
                batch_gen = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen.read_data(args,vid_list_file)

                batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen_tst.read_data(args,vid_list_file_tst)

                trainer.trainG(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

            if args.action == "predict":
                batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen_tst.read_data(args,vid_list_file_tst)
                file_list = "test_split.pkl"
                num_epochs = 40
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 60
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 70
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 80
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 90
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                sample_rate)
                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
    elif mymodel == 'MS_TCN++':
        # print(os.environ["CUDA_VISIBLE_DEVICES"])
        # print(args.goat)
        num_layers_PG = 11
        num_layers_R = 10
        num_R = 3
        num_f_maps=64
        trainer = MSTCN2.Trainer(num_layers_PG, num_layers_R, num_R, num_f_maps, features_dim, num_classes,args)
        if not args.goat == 2:
            if features_dim ==768:
                model_dir = "./{}{}goat/".format(args.model, args.model_dir) + args.dataset + "/split_" + args.split+"/"
                results_dir = "./{}{}goat/".format(args.model, args.result_dir) + args.dataset + "/split_" + args.split+"/"
            else:
                model_dir = "./{}{}goat+i3d/".format(args.model, args.model_dir) + args.dataset + "/split_" + args.split
                results_dir = "./{}{}goat+i3d/".format(args.model, args.result_dir) + args.dataset + "/split_" + args.split+"/"
            if  not os.path.exists(model_dir):
                os.makedirs(model_dir)
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            if args.action == "train":
                batch_gen = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen.read_data(args,vid_list_file)
                trainer.setgoat(vid_list_file,vid_list_file_tst)
                batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen_tst.read_data(args,vid_list_file_tst)
                trainer.trainG(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

            if args.action == "predict":
                batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen_tst.read_data(args,vid_list_file_tst)
                trainer.setgoat(vid_list_file,vid_list_file_tst)
                file_list = "test_split.pkl"
                num_epochs = 40
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 60
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 70
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 80
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 90
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)
                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 100
                trainer.predictG(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir, file_list)

                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
        else:
            if args.action == "train":
                batch_gen = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen.read_data(args,vid_list_file)
                batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen_tst.read_data(args,vid_list_file_tst)

                trainer.train(model_dir, batch_gen, num_epochs, bz, lr, batch_gen_tst)

            if args.action == "predict":
                batch_gen_tst = BatchGenerator(num_classes, gt_path, features_path, sample_rate)
                batch_gen_tst.read_data(args,vid_list_file_tst)
                trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)
                file_list = "test_split.pkl"
                num_epochs = 70
                trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir+'/', file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 80
                trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir+'/', file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 90
                trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)
                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir+'/', file_list)
                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)
                num_epochs = 100
                trainer.predict(model_dir, results_dir, features_path, batch_gen_tst, num_epochs, actions_dict,
                                 sample_rate)

                acc_all, edit_all, f1s_all = func_eval(args.dataset, results_dir+'/', file_list)

                print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)