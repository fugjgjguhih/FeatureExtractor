import pandas as pd
import numpy as np
import pickle as pk
folders =["train","valid","test"]
vidlen_path="vlen.pkl"
vidlens={}
with open(vidlen_path,'rb') as f:
  vidlen_old=pk.load(f)
for k, v in zip(vidlen_old.keys(),vidlen_old.values()):
    vidn = k.split('/')[-1].split('.')[0];
    vidlens[vidn]=v
for folder in folders:
    df = pd.read_csv("{0}.csv".format(folder))
    percentage = 0.2
    annoted ={}
    labels=[]

    for row_index, row in df.iterrows():
        video_n = row[2].split('.')[0]
        count = row[3]
        vidlen = vidlens[video_n]
        print(video_n+" "+str(vidlen))
        actionlabel = np.zeros(int(vidlen))
        for i in range(0,count):
            start = row[i*2+4]
            end  = row[i*2+5]
            action_len = end-start
            start +=action_len*percentage
            end  -=action_len*percentage
            start=int(start)
            end =int(end)
            actionlabel[start:end]=1
        annoted[video_n+".npy"]=[count,actionlabel]
    with open(folder+"annoted.pkl","wb") as f:
        pk.dump(annoted,f)
    with open(vidlen_path,"wb") as f:
        pk.dump(vidlens,f)
