# S2VT + Attention

This is a re-tooled fork of a [discontinued repo by Sundrops](https://github.com/Sundrops/video-caption.pytorch).

The goal is to have something basic for quick video captioning experiments in PyTorch, while still implementing encoder-decoder framework and attention mechanism, and being somewhat updated (having latest PyTorch and CUDA 9 support, for starters).

## News:

[Click here for announced project data (OneDrive)](https://uflorida-my.sharepoint.com/:f:/g/personal/w_garcia_ufl_edu/EuBZsdxZ7BdGva2vP14AlX8BBm4E4uM5SEnu3sogkl_C2g?e=10cxVw)

- `save/`: model checkpoints
- `data/`: preprocessed features for datasets along with their JSON meta files. 

For any model folder, just check `opt_info.json` in the folder for model configuration details. 

Date | Announcement
:----:|:-----------------------------------:|
Nov 8 2018 | Pretrained model and NasNet-A Large features now available for MSVD. Ran for 1000 epochs. Bleu4=0.38.  



# Setup:

If you just want to caption some of your own videos skip to step 5. To train and evaluate on datasets, start at 1.

## 1. Data processing

Borrowed some scripts from [VideoToTextDNN](https://github.com/OSUPCVLab/VideoToTextDNN).

First process frames of MSVD:

```bash
python process_frames.py /path/to/videos /path/to/write/frames/directories 0 n
```

`n = size of dataset`

Next process features, this will take awhile depending on how many GPUs you have (nasnet-large shown):

```bash
python process_features.py /path/to/written/frames /path/to/write/features --type nasnetalarge
```

Batch size parameter `-bs` can be used to adjust batch size in case you run out of memory. The default of 8 uses around 10gb. Smaller batch size = less memory, but will take longer. 

Now process the dataset file. We will use a pkl file from [arctic-capgen-vid](https://github.com/yaoli/arctic-capgen-vid). [Download Link](http://lisaweb.iro.umontreal.ca/transfert/lisa/users/yaoli/youtube2text_iccv15.zip) 
```bash
python process_dataset.py --gtdict /path/to/downloaded/youtube2text_iccv15/dict_movieID_caption.pkl
```

## 2. Patches

Some quick but important tests for coco-caption, the caption eval scorer, before we continue:

Enter `coco-caption/pycocoevalcap/meteor` and try running

```bash
java -jar -Xmx2G meteor-1.5.jar -- -stdio -l en -norm
``` 

You might get an error complaining about a data file. To fix re-download the `.gz` file it expects:
https://github.com/lichengunc/refer/tree/master/evaluation/meteor/data
to `coco-caption/pycocoevalcap/meteor/data/`

If its java related, make sure you have java working first since coco-caption relies on it. 


## 3. Training

## 4. Eval

## 5. Inference
