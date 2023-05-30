import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_info', type=str, help='info that will be displayed when logging', default='Exp1')
    parser.add_argument('--std', type=float, help='standard deviation for gaussian distribution learning', default=5)
    parser.add_argument('--save', action='store_true', help='if set true, save the best model', default=False)
    parser.add_argument('--type', type=str, help='type of the model: USDL or MUSDL', choices=['USDL', 'MUSDL'], default='USDL')
    parser.add_argument('--temporal_aug', type=int, help='the maximum of random temporal shift, ranges from 0 to 6', default=6)
    parser.add_argument('--gpu', type=int, help='id of gpu device(s) to be used', default=1)

    #[AS]
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    # parser.add_argument('--action', default='train')
    parser.add_argument('--action', default='train')
    parser.add_argument('--num_layer', type=int, default=1, help='number of encoder layers')
    parser.add_argument('--linea_dim', type=int, default=256, help='dimension of query and key')
    parser.add_argument('--attn_drop', type=float, default=0.1, help='drop prob of attention layer')
    parser.add_argument('--goat', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, help='number of training epochs', default=150)
    parser.add_argument('--gcn_layers', type=int, default=1, help='number of gcn layers')
    parser.add_argument('--split',type=str,default='3')
    # [BASIC]
    
    parser.add_argument('--train_batch_size', type=int, help='batch size for training phase', default=20)
    parser.add_argument('--test_batch_size', type=int, help='batch size for test phase', default=6)
    parser.add_argument('--seed', type=int, help='manual seed', default=42)
    parser.add_argument('--num_workers', type=int, help='number of subprocesses for dataloader', default=4)
    parser.add_argument('--goat_lr', type=float, help='learning rate', default=3e-4)
    parser.add_argument('--weight_decay', type=float, help='L2 weight decay', default=1e-5)

    # [GOAT SETTING BELOW]
    # [CNN]
    parser.add_argument('--length', type=int, help='length of videos', default=5506)
    parser.add_argument('--img_size', type=tuple, help='input image size', default=(224, 224))
    parser.add_argument('--out_size', type=tuple, help='output image size', default=(25, 25))
    parser.add_argument('--crop_size', type=tuple, help='RoiAlign image size', default=(5, 5))

    # [GCN]
    parser.add_argument('--num_boxes', type=int, help='boxes number of each frames', default=8)
    parser.add_argument('--num_selected_frames', type=int, help='number of selected frames per 16 frames', default=1)

    # [PATH]
    parser.add_argument('--data_path', type=str, help='root of dataset', default='/mnt/disk_1/xiangwei/Bridge-Prompt-master/data/breakfast/frames')
    parser.add_argument('--anno_path', type=str, help='path of annotation file', default='anno_dict.pkl')
    parser.add_argument('--boxes_path', type=str, help='path of boxes annotation file', default='/mnt/disk_1/xiangwei/bbox.pkl')
    # backbone features path
    parser.add_argument('--i3d_feature_path', type=str, help='path of i3d feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/video_feature_dict.pkl')
    parser.add_argument('--swin_feature_path', type=str, help='path of swin feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/video-swin-features/swin_features_dict.pkl')
    parser.add_argument('--bpbb_feature_path', type=str, help='path of bridge-prompt feature dict', default='')
    # attention features path
    parser.add_argument('--cnn_feature_path', type=str, help='path of cnn feature dict', default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/Inceptionv3/inception_feature_dict.pkl')
    parser.add_argument('--bp_feature_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/bp_features', help='bridge prompt feature path')
    parser.add_argument('--formation_feature_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/AS-AQA/formation_features_middle_1.pkl', help='formation feature path')
    # others
    parser.add_argument('--stage1_model_path', type=str, default='/mnt/petrelfs/daiwenxun/AS-AQA/Exp/Group-AQA-Distributed/ckpts/STAGE1_256frames_rho0.3257707338254451_(224, 224)_(25, 25)_loss82.48323059082031.pth', help='stage1_model_path')
    parser.add_argument('--train_dropout_prob', type=float, default=0.3, help='train_dropout_prob')

    # [BOOL]
    # bool for attention mode[GOAT / BP / FORMATION / SELF]
    parser.add_argument('--use_goat', type=int, help='whether to use group-aware-attention', default=1)
    parser.add_argument('--use_bp', type=int, help='whether to use bridge prompt features', default=0)
    parser.add_argument('--use_formation', type=int, help='whether to use formation features', default=0)
    parser.add_argument('--use_self', type=int, help='whether to use self attention', default=0)
    # bool for backbone[I3D / SWIN / BP]
    parser.add_argument('--use_i3d_bb', type=int, help='whether to use i3d as backbone', default=1)
    parser.add_argument('--use_swin_bb', type=int, help='whether to use swin as backbone', default=0)
    parser.add_argument('--use_bp_bb', type=int, help='whether to use bridge-prompt as backbone', default=0)
    # bool for others
    parser.add_argument('--train_backbone', type=int, help='whether to train backbone', default=0)
    parser.add_argument('--use_gcn', type=int, help='whether to use gcn', default=1)
    parser.add_argument('--warmup', type=int, help='whether to warm up', default=1)
    parser.add_argument('--random_select_frames', type=int, help='whether to select frames randomly', default=1)
    parser.add_argument('--use_multi_gpu', type=int, help='whether to use multi gpus', default=1)
    parser.add_argument('--gcn_temporal_fuse', type=int, help='whether to fuse temporal node before gcn', default=0)
    parser.add_argument('--use_cnn_features', type=int, help='whether to use pretrained cnn features', default=1)

    # [LOG]
    parser.add_argument('--exp_name', type=str, default='goat', help='experiment name')
    parser.add_argument('--result_path', type=str, default='result/result.csv', help='result log path')

    # [ATTENTION]
    parser.add_argument('--num_heads', type=int, default=8, help='number of self-attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='number of encoder layers')
    parser.add_argument('--linear_dim', type=int, default=256, help='dimension of query and key')

    # [FIXED PARAMETERS]
    parser.add_argument('--emb_features', type=int, default=1056, help='output feature map channel of backbone')
    parser.add_argument('--num_features_boxes', type=int, default=1024, help='dimension of features of each box')
    parser.add_argument('--num_features_relation', type=int, default=256, help='dimension of embedding phi(x) and theta(x) [Embedded Dot-Product]')
    parser.add_argument('--num_features_gcn', type=int, default=1024, help='dimension of features of each node')
    parser.add_argument('--num_graph', type=int, default=16, help='number of graphs')

    parser.add_argument('--pos_threshold', type=float, default=0.2, help='threshold for distance mask')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)


    parser.add_argument("--date", help="today", type=str, default='0804')
    parser.add_argument('--model', default='MS_TCN++')

    parser.add_argument('--dataset', default="breakfast")
    parser.add_argument('--model_dir', default='models')
    parser.add_argument('--result_dir', default='results')
    parser.add_argument('--epochs', default=90, type=int)
    parser.add_argument("--num", help="how many times do you train today? It is only used for TensorBoardX", type=str,
                        default='1')
    parser.add_argument('--num_stages', default=4, type=int, help='stage number')
    parser.add_argument('--num_f_maps', default=64, type=int, help='embedded feat. dim.')
    parser.add_argument('--features_dim', default=1024, type=int, help='input feat. dim.')
    parser.add_argument('--DA_adv', default='none', type=str, help='adversarial loss (none | rev_grad)')
    parser.add_argument('--DA_adv_video', default='none', type=str,
                        help='video-level adversarial loss (none | rev_grad | rev_grad_ssl | rev_grad_ssl_2)')
    parser.add_argument('--pair_ssl', default='all', type=str, help='pair-feature methods for SSL-DA (all | adjacent)')
    parser.add_argument('--num_seg', default=10, type=int, help='segment number for each video')
    parser.add_argument('--place_adv', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                        metavar='N', help='len(place_adv) == num_stages')
    parser.add_argument('--multi_adv', default=['N', 'N'], type=str, nargs="+",
                        metavar='N', help='separate weights for domain discriminators')
    parser.add_argument('--weighted_domain_loss', default='Y', type=str,
                        help='weighted domain loss for class-wise domain discriminators')
    parser.add_argument('--ps_lb', default='soft', type=str, help='pseudo-label type (soft | hard)')
    parser.add_argument('--source_lb_weight', default='pseudo', type=str,
                        help='label type for source data weighting (real | pseudo)')
    parser.add_argument('--method_centroid', default='none', type=str,
                        help='method to get centroids (none | prob_hard)')
    parser.add_argument('--DA_sem', default='mse', type=str, help='metric for semantic loss (none | mse)')
    parser.add_argument('--place_sem', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                        metavar='N', help='len(place_sem) == num_stages')
    parser.add_argument('--ratio_ma', default=0.7, type=float, help='ratio for moving average centroid method')
    parser.add_argument('--DA_ent', default='none', type=str, help='entropy-related loss (none | target | attn)')
    parser.add_argument('--place_ent', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                        metavar='N', help='len(place_ent) == num_stages')
    parser.add_argument('--use_attn', type=str, default='none', choices=['none', 'domain_attn'],
                        help='attention mechanism')
    parser.add_argument('--DA_dis', type=str, default='none', choices=['none', 'JAN'], help='discrepancy method for DA')
    parser.add_argument('--place_dis', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                        metavar='N', help='len(place_dis) == num_stages')
    parser.add_argument('--DA_ens', type=str, default='none', choices=['none', 'MCD', 'SWD'],
                        help='ensemble method for DA')
    parser.add_argument('--place_ens', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                        metavar='N', help='len(place_ens) == num_stages')
    parser.add_argument('--SS_video', type=str, default='none', choices=['none', 'VCOP'],
                        help='video-based self-supervised learning method')
    parser.add_argument('--place_ss', default=['N', 'Y', 'Y', 'N'], type=str, nargs="+",
                        metavar='N', help='len(place_ss) == num_stages')
    # config & setting
    parser.add_argument('--path_data', default='data/')
    parser.add_argument('--path_model', default='models/')
    parser.add_argument('--path_result', default='results/')
    parser.add_argument('--use_target', default='none', choices=['none', 'uSv'])
    parser.add_argument('--split_target', default='0', help='split for target data (0: no additional split for target)')
    parser.add_argument('--ratio_source', default=1, type=float,
                        help='percentage of total length to use for source data')
    parser.add_argument('--ratio_label_source', default=1, type=float,
                        help='percentage of labels to use for source data (after previous processing)')
    # hyper-parameters
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--bS', default=1, type=int, help='batch size')
    parser.add_argument('--alpha', default=0.15, type=float, help='weighting for smoothing loss')
    parser.add_argument('--tau', default=4, type=float, help='threshold to truncate smoothing loss')
    parser.add_argument('--beta', default=[-2, -2], type=float, nargs="+", metavar='M',
                        help='weighting for adversarial loss & ensemble loss ([frame-beta, video-beta])')
    parser.add_argument('--iter_max_beta', default=[1000, 1000], type=float, nargs="+", metavar='M',
                        help='for adaptive beta ([frame-beta, video-beta])')
    parser.add_argument('--gamma', default=-2, type=float, help='weighting for semantic loss')
    parser.add_argument('--iter_max_gamma', default=1000, type=float, help='for adaptive gamma')
    parser.add_argument('--mu', default=1, type=float, help='weighting for entropy loss')
    parser.add_argument('--nu', default=-2, type=float, help='weighting for the discrepancy loss')
    parser.add_argument('--eta', default=1, type=float, help='weighting for the self-supervised loss')
    parser.add_argument('--iter_max_nu', default=1000, type=float, metavar='M', help='for adaptive nu')
    parser.add_argument('--dim_proj', default=128, type=int, help='projection dimension for SWD')
    # runtime
    parser.add_argument('--verbose', default=False, action="store_true")
    parser.add_argument('--use_best_model', type=str, default='none', choices=['none', 'source', 'target'],
                        help='save best model')
    parser.add_argument('--multi_gpu', default=False, action="store_true")
    parser.add_argument('--resume_epoch', default=0, type=int)
    # tensorboard
    parser.add_argument('--use_tensorboard', default=False, action='store_true')
    parser.add_argument('--epoch_embedding', default=50, type=int,
                        help='select epoch # to save embedding (-1: all epochs)')
    parser.add_argument('--stage_embedding', default=-1, type=int,
                        help='select stage # to save embedding (-1: last stage)')
    parser.add_argument('--num_frame_video_embedding', default=50, type=int,
                        help='number of sample frames per video to store embedding')
    parser.add_argument('--feature', default='Swin')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args
