import pickle

import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from Gconfig  import  *
def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content
 
 
def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i)
    return labels, starts, ends
 
 
def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
 
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
     
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
 
    return score
 
 
def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)
 
 
def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)
 
    tp = 0
    fp = 0
 
    hits = np.zeros(len(y_label))
 
    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()
 
        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)
 
def segment_bars(save_path, *labels):
    num_pics = len(labels)
    color_map = plt.get_cmap('seismic')
    # color_map =
    fig = plt.figure(figsize=(15, num_pics * 1.5))
 
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0, vmax=20)
 
    for i, label in enumerate(labels):
        plt.subplot(num_pics, 1,  i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow([label], **barprops)
 
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
 
    plt.close()
 
 
def segment_bars_with_confidence(save_path, confidence, *labels):
    num_pics = len(labels) + 1
    color_map = plt.get_cmap('seismic')
 
    axprops = dict(xticks=[], yticks=[], frameon=False)
    barprops = dict(aspect='auto', cmap=color_map,
                    interpolation='nearest', vmin=0)
    fig = plt.figure(figsize=(15, num_pics * 1.5))
 
    interval = 1 / (num_pics+1)
    for i, label in enumerate(labels):
        i = i + 1
        ax1 = fig.add_axes([0, 1-i*interval, 1, interval])
        ax1.imshow([label], **barprops)
 
    ax4 = fig.add_axes([0, interval, 1, interval])
    ax4.set_xlim(0, len(confidence))
    ax4.set_ylim(0, 1)
    ax4.plot(range(len(confidence)), confidence)
    ax4.plot(range(len(confidence)), [0.3] * len(confidence), color='red', label='0.5')
 
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()
 
    plt.close()
 
 
def func_eval(dataset, recog_path, file_list):
    with open(file_list, 'rb') as f:
        ls = pickle.load(f)
    with open('anno_dict.pkl', 'rb') as f:
        gts = pickle.load(f)
    list_of_videos = []
    for vid in ls:
        if vid in gts:
            list_of_videos.append(vid)
    gt_contents = [gts[vid][-1] for vid in list_of_videos]
    lss = []
    for vid in list_of_videos:
        ss = vid[-2] + '_' + str(vid[-1])+'.mp4'
        lss.append(ss)
    list_of_videos = lss

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
 
    correct = 0
    total = 0
    edit = 0
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
    index2dict=dict(zip(actions_dict.values(),actions_dict.keys()))
    for idx,vid in enumerate(list_of_videos):
        recog_file = recog_path + vid.split('.')[0]
        recog_content = read_file(recog_file).split('\n')[1].split()
        rec = []
        i=0
        for d in range(len(recog_content)):
            if i>=len(recog_content): break
            if recog_content[i]=='Required':
                rec.append(recog_content[i]+' '+recog_content[i+1]+' '+ recog_content[i+2])
                i+=2
            elif recog_content[i] == 'Acrobatic' or recog_content[i] == 'Cadence':
                rec.append(recog_content[i]+' '+recog_content[i+1])
                i+=1
            else:
                rec.append(recog_content[i])
            i+=1
        gt_content=gt_contents[idx]
        gt_content=[index2dict[value-1] for value in gt_content ]
        recog_content = rec
        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1

        edit += edit_score(recog_content, gt_content)
 
        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
     
     
    acc = 100 * float(correct) / total
    edit = (1.0 * edit) / len(list_of_videos)
#     print("Acc: %.4f" % (acc))
#     print('Edit: %.4f' % (edit))
    f1s = np.array([0, 0 ,0], dtype=float)
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])
 
        f1 = 2.0 * (precision * recall) / (precision + recall)
 
        f1 = np.nan_to_num(f1) * 100
#         print('F1@%0.2f: %.4f' % (overlap[s], f1))
        f1s[s] = f1
 
    return acc, edit, f1s

def main():
    cnt_split_dict = {
        '50salads':5,
        'gtea':4,
        'breakfast':1
    }

    args = get_parser()

    acc_all = 0.
    edit_all = 0.
    f1s_all = [0.,0.,0.]
    model = args.model
    if args.split == 0:
        for split in range(1, cnt_split_dict[args.dataset] + 1):
            print('split')
            recog_path = "./{}/".format(args.result_dir)+args.dataset+"/split_{}".format(split)+"/"
            file_list = "test_split.pkl"
            acc, edit, f1s = func_eval(args.dataset, recog_path, file_list)
            acc_all += acc
            edit_all += edit
            f1s_all[0] += f1s[0]
            f1s_all[1] += f1s[1]
            f1s_all[2] += f1s[2]
        
        acc_all /=  cnt_split_dict[args.dataset]
        edit_all /= cnt_split_dict[args.dataset]
        f1s_all = [i / cnt_split_dict[args.dataset] for i in f1s_all]
    else:
        recog_path = "./{}/".format(args.result_dir) + args.dataset + "/split_{}".format(0) + "/"  # 改成split 0了
        if model == 'BP':
            recog_path = "./{}/".format(args.result_dir) + args.dataset + "/split_{}".format(0) + "/"  # 改成split 0了
        elif model == 'ASFormer':
            recog_path = "./I3D{}/".format(args.result_dir) + args.dataset + "/split_{}".format(0) + "/"  # 改成split 0了
        elif model == 'MS_TCN++':
            recog_path = "./MS_TCN++{}/".format(args.result_dir) + args.dataset + "/split_{}".format(args.split) + "/"  # 改成split 0了
            if  args.goat == 0:
                if args.feature == 'I3D':
                    recog_path = "./MS_TCN++{}goat+i3d/".format(args.result_dir) + args.dataset + "/split_{}".format(args.split) + "/"
                if args.feature == 'BP':
                    recog_path = "./MS_TCN++{}goat/".format(args.result_dir) + args.dataset + "/split_{}".format(args.split) + "/"
            else:
                    recog_path = "./MS_TCN++{}{}/".format(args.result_dir,args.feature) + args.dataset + "/split_{}".format(args.split) + "/"

        elif model == 'SSTDA':
            if not args.goat == 2:
                recog_path = "./SSTDA{}goatresults/".format(args.feature) + args.dataset + "/split_{}".format(args.split) + "/"
            else:
                recog_path = "./SSTDA{}{}/".format(args.feature,args.result_dir) + args.dataset + "/split_{}".format(args.split) + "/"
        file_list = "test_split.pkl"
        print('Feature:'+args.feature +'   lr:'+str(args.lr) +'   '+recog_path)
        acc_all, edit_all, f1s_all = func_eval(args.dataset, recog_path, file_list)
    
    print("Acc: %.4f  Edit: %4f  F1@10,25,50 " % (acc_all, edit_all), f1s_all)



if __name__ == '__main__':
    main()