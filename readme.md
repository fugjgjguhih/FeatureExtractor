## Introduction

This repository provide a method  to  extract  a frame-wise RGB feature for video analyses.  It split video into clips by taking the contextual frame of each frame (the several frame behind and ahead) , than fed these splits into the 3D backbone.

## Installation

My code has been implemented on Python 3.9 and PyTorch 1.8.1+cu116 ,See other required packages in `requirements.txt`.

than install this repository  [hassony2/torch_videovision: Transforms for video datasets in pytorch (github.com)](https://github.com/hassony2/torch_videovision)

```
``pip install -r requirements.txt`
```

## Data preparation

Firstly,  you should download pre-trained ckpts of backbones from their sites  and place them to ./checkpoint

Secondly, you should split the  raw videos into 3-folds(train,test,vaild) or you can modify the folders list in  vid2frame.py and feature extractor,  you can use the ./utils/vid2frame.py  to divide video into frames and get the dict of video-length(vlen.pkl)

```
python ./utils/vid2frame.py --vpath=${RAW_VIDEO_PATH}  --fpath=${OUTPUT_FRAME_PATH}
```

Third, put the vlen.pkl  in the working path.

## Usage

#### Extract feature

Currently, the repository provide method to extract swin and RGB-I3D

```
python SwinFeature.py --vpath=${VIDEO_FRAME_PATH}  --spath=${FEATURE_SAVE_PATH} --scale=${SIZE_OF_EACH_CLIPS}
or
python I3DFeature.py --vpath=${VIDEO_FRAME_PATH}  --spath=${FEATURE_SAVE_PATH} --scale=${SIZE_OF_EACH_CLIPS}
```

#### Expendition

You can add the backbone network code in the   ./model fold.   than use it instance to the backbone  variable in SwinFeature.py  and change the feature_dim

#### Use for Action Segmentation

This repository also provide a sample  implement of  ASFormer, MS_TCN++,SSTDA, whose official code only support the Breakfast, GTEA50. and you can use this repository to extract feature of you dataset.

- You can transform the  human action counting dataset using .utils/count2segment.py
- You can use the ./sample/AS/dataset/batch_gen.py   as a dataloader to load the fine-grained labels and features,  it is modify from the ASFormer's  officical implemental.
-  You can run  ./sample/AS/AS.py  for MS_TCN++  and ASFormer  



