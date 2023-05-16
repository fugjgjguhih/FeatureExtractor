import os
import pickle
import cv2
from tqdm import tqdm
import os
import glob
import argparse
from sklearn.model_selection import  train_test_split
parser = argparse.ArgumentParser()
parser.add_argument('--fps', default=30)
parser.add_argument('--vpath', default='/mnt/disk_1/xiangwei/LL/video')
parser.add_argument('--fpath', default='/mnt/disk_1/xiangwei/LL/frame')
args = parser.parse_args()
path =args.vpath
output = args.fpath
fps = args.fps
# path =r"C:\Users\LOA\Desktop\test\v"
# output= r"C:\Users\LOA\Desktop\test\f"
file_suffix = ['*.mp4', '*.avi', '*.webm']
dirpaths=[]
file_list=[]

for dirpath, dirnames, filenames in os.walk(path):
    #if not dirnames  最低层
    if not dirnames:
        files=[]
        for s in file_suffix:
            files.extend(glob.glob(os.path.join(dirpath, '**', s), recursive=True))
        files_prefix = os.path.commonprefix(files)
        files_prefix = files_prefix.rsplit('/', 1)[0]
        file_list.append(files)
vv= {}

def video2frame(videos_path, frames_save_path, time_interval):
    '''
    :param videos_path: 视频的存放路径
    :param frames_save_path: 视频切分成帧之后图片的保存路径
    :param time_interval: 保存间隔
    :return:
    '''
    vidcap = cv2.VideoCapture(videos_path)
    success, image = vidcap.read()
    count = 0
    if not os.path.exists(frames_save_path):
       os.makedirs(frames_save_path)
    while success:
        success, image = vidcap.read()

        if image is None:
            vv[videos_path]=count;
            break
        count += 1
        cv2.imwrite(os.path.join(frames_save_path,"%d.jpg" %count), image)
        # if count % time_interval == 0:
        #     cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/%d.jpg" % count)
for video_list in file_list:
    for v in tqdm(video_list):
         vpath = v.split('/')
         foldnames, vidn = vpath[-2],vpath[-1];
         savepath = os.path.join(output,foldnames,vidn.split('.')[-2])
         print(v)
         video2frame(v,savepath,1)
with open('vlen.pkl','wb') as f:
    pickle.dump(vv,f)