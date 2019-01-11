import argparse
import torch

from utils import *
from ImageCaptioner import ImageCaptioner
from yunjey_image_captioning.build_vocab import Vocabulary
from torchvision import transforms

from attack import CarliniAttack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
python _unittest_image_captioner.py
--image yunjey_image_captioning/png/example.png 
--encoder_path /path/to/encoder-5-3000.pkl 
--decoder_path /path/to/decoder-5-3000.pkl 
--vocab_path /path/to/vocab.pkl
"""

#python _unittest_image_captioner.py --image yunjey_image_captioning/png/example.png --encoder_path C:\Users\Shumpu\Documents\encoder-5-3000.pkl --decoder_path C:\Users\Shumpu\Documents\decoder-5-3000.pkl --vocab_path C:\Users\Shumpu\Documents\vocab.pkl
#python _unittest_carliniattack.py --image yunjey_image_captioning/png/example.png --encoder_path C:\Users\Shumpu\Documents\encoder-5-3000.pkl --decoder_path C:\Users\Shumpu\Documents\decoder-5-3000.pkl --vocab_path C:\Users\Shumpu\Documents\vocab.pkl
#python _unittest_carliniattack.py --image yunjey_image_captioning/png/example2.jpeg --encoder_path C:\Users\Shumpu\Documents\encoder-5-3000.pkl --decoder_path C:\Users\Shumpu\Documents\decoder-5-3000.pkl --vocab_path C:\Users\Shumpu\Documents\vocab.pkl

def main(args):
    # Prepare an image
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406),
                                                         (0.229, 0.224, 0.225))])

    image = load_image(args.image, transform)
    image_tensor = image.to(device)

    #a group of zebra standing next to each other on a sunny day .
    #a bird sitting on a wooden table with a bird .

    im_captioner = ImageCaptioner(args)
    chosen_caption = "a bird sitting on a wooden table with a bird ."
    # loss = im_captioner.forward(image_tensor, chosen_caption)
    print(im_captioner.caption_file(args.image))

    carlini = CarliniAttack(oracle=im_captioner, image = image, target=chosen_caption)

    carlini.execute(args.image, args.image)

    # print(loss.item())
    # print(.shape)



##TO DO: Add an option to send a target image, for which it will generate a caption and use that instead of you manually passing in a caption.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--input', type=str, nargs='+', required=True, help="Input .jpeg images, seperated by spaces")
    #.add_argument('--target', type=str, required=True, help="Target transcription for images.")
    #parser.add_argument('--out', required=True, help="Directory to save adversarial examples.")
   # parser.add_argument('--ckpt', required=True, help="Path to the best trained netCRNN .pth file.")
    #parser.add_argument('--alphabet', required=True, help="Path to alphabet of the trained checkpoint.")
    #parser.add_argument('--cuda', action='store_true', default=False, help="Use CUDA.")
    #parser.add_argument('--seed', default=9, help="Random seed to use.")
    #args = parser.parse_args()

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
    #parser.add_argument('--target', type=str, required=False, 'help='input image for target caption')

    '''
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
    '''
    args = parser.parse_args()

    main(args)

    #
    # _validate(args)
    #
    # torch.manual_seed(args.seed)
    #
    # try:
    #     with io.open(args.alphabet, 'r', encoding='utf-8') as myfile:
    #         alphabet = myfile.read().split()
    #         alphabet.append(u' ')
    #         alphabet = ''.join(alphabet)
    #
    #     converter = utils.strLabelConverter(alphabet, attention=False)
    #
    #     nclass = converter.num_classes
    #
    #     crnn = models.crnn.CRNN(imgH, nc, nclass, num_hidden)
    #     crnn.apply(weights_init)
    #
    #     if args.cuda:
    #         crnn = crnn.cuda()
    #         crnn = torch.nn.DataParallel(crnn)
    #
    #     logger.info("Loading pretrained model from {}".format(args.ckpt))
    #     file_weights = torch.load(args.ckpt)
    #
    #     crnn.load_state_dict(file_weights)
    #
    #     print("The oracle network:", crnn)  # Logging can't print torch models :thinking:
    #
    #     image = Image.open(args.input[0]).convert('L')
    #     attack = CarliniAttack(crnn, alphabet, image.size, args.target, file_weights)
    #
    #     attack.execute(args.input, args.out)
    #
    # except KeyboardInterrupt:
    #
    #     pass