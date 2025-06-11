import os
import json
import h5py
import torch
import imageio
import cv2
import numpy as np
from collections import Counter
from random import seed, choice, sample
from tqdm import tqdm


def imread(path):
    """
    Reads an image from disk and returns it as a numpy array (RGB).
    """
    img = imageio.imread(path)
    if img.ndim == 2:  # grayscale to RGB
        img = np.stack([img] * 3, axis=2)
    elif img.shape[2] == 4:  # RGBA to RGB
        img = img[:, :, :3]
    return img


def imresize(image, size):
    """
    Resizes an image to the specified size (width, height) using bilinear interpolation.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)


def create_input_files(dataset, karpathy_json_path, image_folder, captions_per_image,
                       min_word_freq, output_folder, max_len=100):
    """
    Creates input files for training, validation, and test data.

    :param dataset: 'coco', 'flickr8k', or 'flickr30k'
    :param karpathy_json_path: Path to Karpathy JSON file with splits and captions
    :param image_folder: Folder with downloaded images
    :param captions_per_image: Number of captions to sample per image
    :param min_word_freq: Minimum frequency of words to be kept in vocabulary
    :param output_folder: Folder to save processed files
    :param max_len: Maximum caption length allowed (captions longer than this are skipped)
    """
    assert dataset in {'coco', 'flickr8k', 'flickr30k'}

    with open(karpathy_json_path, 'r') as j:
        data = json.load(j)

    train_image_paths, train_image_captions = [], []
    val_image_paths, val_image_captions = [], []
    test_image_paths, test_image_captions = [], []
    word_freq = Counter()

    # Gather captions and paths per split
    for img in data['images']:
        captions = []
        for c in img['sentences']:
            if len(c['tokens']) <= max_len:
                captions.append(c['tokens'])
                word_freq.update(c['tokens'])

        if len(captions) == 0:
            continue

        # Construct image path based on dataset
        if dataset == 'coco':
            path = os.path.join(image_folder, img['filepath'], img['filename'])
        else:
            path = os.path.join(image_folder, img['filename'])

        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] == 'val':
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] == 'test':
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)

    # Create word map
    words = [w for w in word_freq if word_freq[w] > min_word_freq]
    word_map = {word: idx + 1 for idx, word in enumerate(words)}
    # Add special tokens
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0

    base_filename = f"{dataset}_{captions_per_image}_cap_per_img_{min_word_freq}_min_word_freq"

    # Save word map
    os.makedirs(output_folder, exist_ok=True)
    with open(os.path.join(output_folder, f'WORDMAP_{base_filename}.json'), 'w') as j:
        json.dump(word_map, j)

    seed(123)  # for reproducibility

    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(output_folder, f'{split}_IMAGES_{base_filename}.hdf5'), 'w') as h:
            h.attrs['captions_per_image'] = captions_per_image
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print(f"\nReading {split} images and captions, storing to file...\n")

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions for this image
                if len(imcaps[i]) < captions_per_image:
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(captions_per_image - len(imcaps[i]))]
                else:
                    captions = sample(imcaps[i], k=captions_per_image)

                assert len(captions) == captions_per_image

                # Read and resize image
                img = imread(path)
                img = imresize(img, (256, 256))
                img = img.transpose(2, 0, 1)  # HWC to CHW
                assert img.shape == (3, 256, 256)
                assert np.max(img) <= 255

                images[i] = img

                for c in captions:
                    # Encode caption with <start> and <end> and pad to max_len
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + \
                            [word_map['<end>']] + [word_map['<pad>']] * (max_len - len(c))
                    c_len = len(c) + 2  # Including <start> and <end>

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            assert images.shape[0] * captions_per_image == len(enc_captions) == len(caplens)

            # Save captions and caption lengths
            with open(os.path.join(output_folder, f'{split}_CAPTIONS_{base_filename}.json'), 'w') as j:
                json.dump(enc_captions, j)

            with open(os.path.join(output_folder, f'{split}_CAPLENS_{base_filename}.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Initializes embedding tensor with uniform distribution.

    :param embeddings: torch.FloatTensor to initialize
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Loads pretrained GloVe embeddings aligned with word_map.

    :param emb_file: Path to GloVe file
    :param word_map: Vocabulary word map
    :return: (embeddings tensor, embedding dimension)
    """
    with open(emb_file, 'r', encoding='utf-8') as f:
        emb_dim = len(f.readline().strip().split(' ')) - 1

    vocab = set(word_map.keys())
    embeddings = torch.FloatTensor(len(word_map), emb_dim)
    init_embedding(embeddings)

    print("\nLoading embeddings...")
    with open(emb_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip().split(' ')
            emb_word = line[0]
            embedding = list(map(float, line[1:]))

            if emb_word in vocab:
                embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients to avoid exploding gradients.

    :param optimizer: optimizer with gradients
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder,
                    encoder_optimizer, decoder_optimizer, bleu4, is_best):
    """
    Saves checkpoint.

    :param data_name: dataset base name
    :param epoch: current epoch
    :param epochs_since_improvement: epochs since last BLEU-4 improvement
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: encoder optimizer
    :param decoder_optimizer: decoder optimizer
    :param bleu4: BLEU-4 score
    :param is_best: flag if this is best model so far
    """
    state = {
        'epoch': epoch,
        'epochs_since_improvement': epochs_since_improvement,
        'bleu-4': bleu4,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'encoder_optimizer_state_dict': encoder_optimizer.state_dict() if encoder_optimizer else None,
        'decoder_optimizer_state_dict': decoder_optimizer.state_dict() if decoder_optimizer else None
    }
    filename = f'checkpoint_{data_name}.pth.tar'
    torch.save(state, filename)

    if is_best:
        torch.save(state, f'BEST_{filename}')


class AverageMeter:
    """
    Computes and stores average and current value of metrics.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Decays learning rate by a given factor.

    :param optimizer: optimizer to update
    :param shrink_factor: factor between 0 and 1
    """
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] *= shrink_factor
    print(f"New learning rate: {optimizer.param_groups[0]['lr']:.6f}\n")


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy.

    :param scores: model output scores (batch_size x num_classes)
    :param targets: true labels (batch_size)
    :param k: top-k
    :return: accuracy percentage
    """
    batch_size = targets.size(0)
    _, ind = scores.topk(k, dim=1, largest=True, sorted=True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)
