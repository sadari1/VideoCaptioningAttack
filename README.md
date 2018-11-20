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

 

