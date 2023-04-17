import os
import pickle
import cv2
from tqdm import tqdm
from sklearn.model_selection import  train_test_split
mydir ='./video'
dirpaths=[]
vid_n=[]
ttt = []
print(1)
for dirpath, dirnames, filenames in os.walk(mydir):
    if not dirnames:
        t = dirpath.split("/")
        s = t[-2] + "_" + t[-1] + ".mp4"
        ttt.append((t[-2],int(t[-1])))
        dirpath+="/"
        dirpaths.append(os.path.abspath(dirpath))
        vid_n.append(s)
vv= {}
x_train,x_test = train_test_split(ttt)
# TODO 修改图片路径、视频路径、标签路径和是否进行label3可视化(路径中不能有中文！！！路径中不能有中文！！！路径中不能有中文！！！)
for path_frames, vid_name in zip(dirpaths, vid_n):
    path_video = os.path.join("/mnt","disk_1","xiangwei","Video",vid_name)
    path_frames=path_frames+"/"
    list_frames = os.listdir(path_frames)
    list_frames.sort(key =lambda x:int(x[:-4]))
     #  print(list_frames)
    fps = 25  # 视频每秒25帧
    size = (1920, 1080)  # 需要转为视频的图片的尺寸
    if(os.path.exists(path_video)):
        vv[vid_name] = len(list_frames)
        print(vid_name)
        print(len(list_frames))
        continue
    video = cv2.VideoWriter(path_video, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    for item in tqdm(list_frames):
        if item.endswith('.jpg'):
            path_item = os.path.join(path_frames, item)
            img = cv2.imread(path_item)
            img = cv2.resize(img, (1920, 1080))
            video.write(img)
    video.release()
    cv2.destroyAllWindows()
with open('vlen.pkl','rb') as f:
    vs= pickle.load(f)
vv.update(vs)
print(vs['Series2022Leg4_tech_3.mp4'])
with open('vlen.pkl','wb') as f:
    pickle.dump(vv,f)
with open('train_split.pkl','wb') as f:
    pickle.dump(x_train,f)
with open('text_split.pkl', 'wb') as f:
    pickle.dump(x_test, f)
with open('vidname.pkl', 'wb') as f:
    pickle.dump(vid_n, f)