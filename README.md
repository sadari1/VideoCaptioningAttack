# VideoCaptioningAttack

Based off of: https://github.com/huanzhang12/ImageCaptioningAttack

### Install: Image Captioner Attack

Clone this repository with:

`git clone --recursive https://github.com/sadari1/VideoCaptioningAttack.git`

`coco/` should appear with contents inside. Assuming you have PyTorch 0.4.0 already installed, proceed:

```bash
cd coco/PythonAPI/
make
python setup.py build
python setup.py install
cd ../../
cd yunjey_image_captioning/
pip install -r requirements.txt
cd ..
```

Now download the image captioner [model](https://www.dropbox.com/s/ne0ixz5d58ccbbz/pretrained_model.zip?dl=0) and [vocab file](https://www.dropbox.com/s/26adb7y9m98uisa/vocap.zip?dl=0) to a memorable place. The image captioning model is taken from [here](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning).

 

### Run Image Captioner

Run this if you want to use the image captioner to caption some image.

```
python _unittest_image_captioner.py
--image yunjey_image_captioning/png/example.png 
--encoder_path /path/to/encoder-5-3000.pkl 
--decoder_path /path/to/decoder-5-3000.pkl 
--vocab_path /path/to/vocab.pkl
```

### Run Image Attack

Run this if  you want to launch the adversarial attack on some image. The chosen caption is preset to "a bird sitting on a wooden table with a bird ." within the main() method. You may change it to a caption of your liking, but unless the sentence is made up of vocabulary tokens that are all within the vocab.pkl, the attack might never converge to that specific caption though it can get close to it.

```
python _unittest_image_attack.py 
--image yunjey_image_captioning/png/example2.jpeg
--encoder_path /path/to/encoder-5-3000.pkl 
--decoder_path /path/to/decoder-5-3000.pkl 
--vocab_path /path/to/vocab.pkl
```

### Run Video Captioner

Run this if you want to launch the video captioner on a specific video.

``` 
python _unittest_videocaptioner.py
"C:/Path/To/Video.avi" 
--recover_opt "C:/Path/To/opt_info.json" 
--saved_model "C:/Path/To/model_1000.pth"
```

### Run Video Attack

Run this if you just want to launch a video attack without saving the adversarial frames.

```
python _unittest_video_attack.py
"C:/Path/To/Directory/InputVideo.avi" 
--recover_opt "C:/Path/To/opt_info.json" 
--saved_model "C:/Path/To/model_1000.pth"
```


Run this if you want to launch a video attack and save the adversarial frames to a specific directory.

```
python experiment_video_attack_rand_test_caption.py
"D:\College\Research\videostoattack" 
"D:\College\Research\attacked" 
--recover_opt "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\opt_info.json" 
--saved_model "D:\College\Research\December 2018 Video Captioning Attack\video captioner\save\msvd_nasnetalarge\model_1000.pth"
```
