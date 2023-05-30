#!/bin/bash
# ----------------------------------------------------------------------------------
# python train.py ./result/50salads/dataset-50salads_split-1/config.yaml
# python train.py ./result/50salads/dataset-50salads_split-1/config1.yaml


# python evaluate.py ./result/50salads/dataset-50salads_split-1/config.yaml
# python evaluate.py ./result/50salads/dataset-50salads_split-1/config1.yaml

# === Mode Switch On/Off === #
split=5
training=true # true | false
predict=true # true | false
eval=true # true | false

# === Paths === #
path_data=../../Datasets/action-segmentation/ # user-defined
path_model=../../Models_SSTDA/action-segmentation-DA/ # user-defined (will create if not existing)
path_result=../../Results_SSTDA/action-segmentation-DA/ # user-defined (will create if not existing)

# === Config & Setting === #
use_target=uSv # none | uSv
dataset=breakfast # gtea | 50salads | breakfast
use_best_model=source # none | source
num_seg=2 # number of segments for each video (SSTDA: 2 | VCOP: 3 | others: 1)

# SSTDA
DA_adv=rev_grad # none | rev_grad
DA_adv_video=rev_grad_ssl # none | rev_grad_ssl
use_attn=domain_attn # none | domain_attn
DA_ent=attn # none | attn

# MADA
multi_adv_2=N # Y | N

# MSTN
method_centroid=none # none | prob_hard

# JAN
DA_dis=none # none | JAN

# MCD / SWD
DA_ens=none # none | MCD | SWD

# VCOP
SS_video=none # none | VCOP

# --- hyper-parameters --- #
iter_max_0=65000
iter_max_1=50000
iter_max_nu=16000000
mu=0.4251
eta=0
lr=0.0003

# === Main Program === #
echo 'use_target: '$use_target', dataset: '$dataset

## run codes ##
for split in 1 
do
    echo 'split: '$split
    # train
    if ($training)
    then
        python SSTDAmain.py --path_data=$path_data --path_model=$path_model --path_result=$path_result \
        --action=train --dataset=$dataset --split=$split --lr $lr --use_target $use_target \
        --DA_adv $DA_adv --DA_adv_video $DA_adv_video --num_seg $num_seg --beta -2 -2 --iter_max_beta $iter_max_0 $iter_max_1 \
        --DA_ent $DA_ent --mu $mu --use_attn $use_attn \
        --multi_adv N $multi_adv_2 \
        --method_centroid $method_centroid --iter_max_gamma $iter_max_0 \
        --DA_dis $DA_dis --iter_max_nu $iter_max_nu \
        --DA_ens $DA_ens \
        --SS_video $SS_video --eta $eta \
        --use_best_model $use_best_model --verbose\
        --feature='I3D'\
        --gpu=$gpu\
        --seed=3353\
        --        
    fi

    # predict
    if ($predict)
    then
        python SSTDAmain.py --path_data=$path_data --path_model=$path_model --path_result=$path_result \
        --action=predict --dataset=$dataset --split=$split --lr $lr  --use_target $use_target \
        --DA_adv_video $DA_adv_video --num_seg $num_seg --use_attn $use_attn \
        --multi_adv N $multi_adv_2 --method_centroid $method_centroid --DA_ens $DA_ens --SS_video $SS_video \
        --use_best_model $use_best_model \
        --feature='I3D'\
        --gpu=$gsplit\
        --linea_dim=$linea_dim\
        --goat_lr=$glr\
        --seed=3353
    fi
            
    # eval
    if ($eval)
    then
        python eval.py --model='SSTDA' --feature='I3D' --split=$split
    fi
done
# training=true # true | false
# predict=true # true | false
# eval=true # true | false

# # === Paths === #
# path_data=../../Datasets/action-segmentation/ # user-defined
# path_model=../../Models_SSTDA/action-segmentation-DA/ # user-defined (will create if not existing)
# path_result=../../Results_SSTDA/action-segmentation-DA/ # user-defined (will create if not existing)

# # === Config & Setting === #
# use_target=uSv # none | uSv
# dataset=breakfast # gtea | 50salads | breakfast
# use_best_model=source # none | source
# num_seg=2 # number of segments for each video (SSTDA: 2 | VCOP: 3 | others: 1)

# # SSTDA
# DA_adv=rev_grad # none | rev_grad
# DA_adv_video=rev_grad_ssl # none | rev_grad_ssl
# use_attn=domain_attn # none | domain_attn
# DA_ent=attn # none | attn

# # MADA
# multi_adv_2=N # Y | N

# # MSTN
# method_centroid=none # none | prob_hard

# # JAN
# DA_dis=none # none | JAN

# # MCD / SWD
# DA_ens=none # none | MCD | SWD

# # VCOP
# SS_video=none # none | VCOP

# # --- hyper-parameters --- #
# iter_max_0=65000
# iter_max_1=50000
# iter_max_nu=16000000
# mu=0.4251
# eta=0
# lr=0.0003

# # === Main Program === #
# echo 'use_target: '$use_target', dataset: '$dataset

# ## run codes ##
# for split in 1 
# do
#     echo 'split: '$split
#     # train
#     if ($training)
#     then
#         python SSTDAmain.py --path_data=$path_data --path_model=$path_model --path_result=$path_result \
#         --action=train --dataset=$dataset --split=$split --lr $lr --use_target $use_target \
#         --DA_adv $DA_adv --DA_adv_video $DA_adv_video --num_seg $num_seg --beta -2 -2 --iter_max_beta $iter_max_0 $iter_max_1 \
#         --DA_ent $DA_ent --mu $mu --use_attn $use_attn \
#         --multi_adv N $multi_adv_2 \
#         --method_centroid $method_centroid --iter_max_gamma $iter_max_0 \
#         --DA_dis $DA_dis --iter_max_nu $iter_max_nu \
#         --DA_ens $DA_ens \
#         --SS_video $SS_video --eta $eta \
#         --use_best_model $use_best_model --verbose\
#         --feature='I3D'\
#         --gpu=$gsplit\
#         --linea_dim=$linea_dim\
#         --goat_lr=$glr\
#         --seed=3353
#     fi

#     # predict
#     if ($predict)
#     then
#         python SSTDAmain.py --path_data=$path_data --path_model=$path_model --path_result=$path_result \
#         --action=predict --dataset=$dataset --split=$split --lr $lr  --use_target $use_target \
#         --DA_adv_video $DA_adv_video --num_seg $num_seg --use_attn $use_attn \
#         --multi_adv N $multi_adv_2 --method_centroid $method_centroid --DA_ens $DA_ens --SS_video $SS_video \
#         --use_best_model $use_best_model \
#         --feature='I3D'\
#         --gpu=$gsplit\
#         --linea_dim=$linea_dim\
#         --goat_lr=$glr\
#         --seed=3353
#     fi
            
#     # eval
#     if ($eval)
#     then
#         python eval.py --model='SSTDA' --feature='I3D' --split=$gsplit
#     fi
# done
# training=true # true | false
# predict=true # true | false
# eval=true # true | false

# # === Paths === #
# path_data=../../Datasets/action-segmentation/ # user-defined
# path_model=../../Models_SSTDA/action-segmentation-DA/ # user-defined (will create if not existing)
# path_result=../../Results_SSTDA/action-segmentation-DA/ # user-defined (will create if not existing)

# # === Config & Setting === #
# use_target=uSv # none | uSv
# dataset=breakfast # gtea | 50salads | breakfast
# use_best_model=source # none | source
# num_seg=2 # number of segments for each video (SSTDA: 2 | VCOP: 3 | others: 1)

# # SSTDA
# DA_adv=rev_grad # none | rev_grad
# DA_adv_video=rev_grad_ssl # none | rev_grad_ssl
# use_attn=domain_attn # none | domain_attn
# DA_ent=attn # none | attn

# # MADA
# multi_adv_2=N # Y | N

# # MSTN
# method_centroid=none # none | prob_hard

# # JAN
# DA_dis=none # none | JAN

# # MCD / SWD
# DA_ens=none # none | MCD | SWD

# # VCOP
# SS_video=none # none | VCOP

# # --- hyper-parameters --- #
# iter_max_0=65000
# iter_max_1=50000
# iter_max_nu=16000000
# mu=0.4251
# eta=0
# lr=0.0003

# # === Main Program === #
# echo 'use_target: '$use_target', dataset: '$dataset

# ## run codes ##
# for split in 1 
# do
#     echo 'split: '$split
#     # train
#     if ($training)
#     then
#         python SSTDAmain.py --path_data=$path_data --path_model=$path_model --path_result=$path_result \
#         --action=train --dataset=$dataset --split=$split --lr $lr --use_target $use_target \
#         --DA_adv $DA_adv --DA_adv_video $DA_adv_video --num_seg $num_seg --beta -2 -2 --iter_max_beta $iter_max_0 $iter_max_1 \
#         --DA_ent $DA_ent --mu $mu --use_attn $use_attn \
#         --multi_adv N $multi_adv_2 \
#         --method_centroid $method_centroid --iter_max_gamma $iter_max_0 \
#         --DA_dis $DA_dis --iter_max_nu $iter_max_nu \
#         --DA_ens $DA_ens \
#         --SS_video $SS_video --eta $eta \
#         --use_best_model $use_best_model --verbose\
#         --feature='swin'
#     fi

#     # predict
#     if ($predict)
#     then
#         python SSTDAmain.py --path_data=$path_data --path_model=$path_model --path_result=$path_result \
#         --action=predict --dataset=$dataset --split=$split --lr $lr  --use_target $use_target \
#         --DA_adv_video $DA_adv_video --num_seg $num_seg --use_attn $use_attn \
#         --multi_adv N $multi_adv_2 --method_centroid $method_centroid --DA_ens $DA_ens --SS_video $SS_video \
#         --use_best_model $use_best_model \
#         --feature='swin'
#     fi
            
#     # eval
#     if ($eval)
#     then
#         python eval.py --model='SSTDA' --feature='swin'
#     fi
# done
# python train.py ./result/50salads/dataset-50salads_split-1/config2.yaml
# python evaluate.py ./result/50salads/dataset-50salads_split-1/config2.yaml
# #----------------------------------------------------------------------------------
# exit 0

