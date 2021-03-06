import argparse
import torch

from utils import *
from ImageCaptioner import ImageCaptioner
from yunjey_image_captioning.build_vocab import Vocabulary
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
python _unittest_image_captioner.py
--image yunjey_image_captioning/png/example.png 
--encoder_path /path/to/encoder-5-3000.pkl 
--decoder_path /path/to/decoder-5-3000.pkl 
--vocab_path /path/to/vocab.pkl
"""

#python _unittest_image_captioner.py --image yunjey_image_captioning/png/example2.jpeg --encoder_path C:\Users\Shumpu\Documents\encoder-5-3000.pkl --decoder_path C:\Users\Shumpu\Documents\decoder-5-3000.pkl --vocab_path C:\Users\Shumpu\Documents\vocab.pkl

def main(args):
    # Prepare an image
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])

    image = load_image(args.image, transform)
    image_tensor = image.to(device)

    im_captioner = ImageCaptioner(args)
    chosen_caption = "my caption"
    # loss = im_captioner.forward(image_tensor, chosen_caption)
    print(im_captioner.caption_file(args.image))
   # print(im_captioner.encoder(image_tensor))
   #  print(loss.item())
    # print(.shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='input image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='models/encoder-2-1000.ckpt',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-2-1000.ckpt',
                        help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')

    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int, default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int, default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int, default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
