import torch
import pickle
import numpy as np
from utils import *
from torchvision import transforms
from yunjey_image_captioning.data_loader import collate_fn
from yunjey_image_captioning.model import EncoderCNN, DecoderRNN
from yunjey_image_captioning.build_vocab import Vocabulary
from torch.nn.utils.rnn import pack_padded_sequence

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ImageCaptioner:
    def __init__(self, args):
        # Image preprocessing
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406),
                                                                  (0.229, 0.224, 0.225))])
        # Load vocabulary wrapper
        with open(args.vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)

        # Build models
        # encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
        encoder = EncoderCNN(args.embed_size)  # train mode (batchnorm uses moving mean/variance)
        decoder = DecoderRNN(args.embed_size, args.hidden_size, len(self.vocab), args.num_layers)
        encoder = encoder.to(device)
        decoder = decoder.to(device)

        # Load the trained model parameters
        encoder.load_state_dict(torch.load(args.encoder_path))
        decoder.load_state_dict(torch.load(args.decoder_path))

        self.encoder = encoder
        self.decoder = decoder

        self.crit = torch.nn.CrossEntropyLoss()

    def forward(self, image_tensor, chosen_caption):
        vocab = self.vocab

        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in chosen_caption.split(' ')])
        caption.append(vocab('<end>'))
        caption = torch.Tensor(caption)

        images, captions, lengths = collate_fn([
            (image_tensor[0], caption)
        ])
        images = images.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        features = self.encoder(images)
        outputs = self.decoder(features, captions, lengths)
        loss = self.crit(outputs, targets)

        return loss

    def caption_file(self, image_path):
        # Prepare an image
        image = load_image(image_path, self.transform)
        image_tensor = image.to(device)

        # Generate an caption from the image
        feature = self.encoder(image_tensor)
        sampled_ids = self.decoder.sample(feature)
        sampled_ids = sampled_ids[0].cpu().numpy()  # (1, max_seq_length) -> (max_seq_length)

        # Convert word_ids to words
        sampled_caption = []
        for word_id in sampled_ids:
            word = self.vocab.idx2word[word_id]
            sampled_caption.append(word)
            if word == '<end>':
                break
        sentence = ' '.join(sampled_caption)

        return sentence
