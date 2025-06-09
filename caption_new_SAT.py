import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to word map JSON file in Kaggle
word_map_path = "/kaggle/working/word_map.json"

# Load word map
with open(word_map_path, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}  # index-to-word mapping

def load_image(image_path):
    """Loads and preprocesses an image."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def caption_image_beam_search(encoder, decoder, image_path, beam_size=3):
    """Generates a caption using beam search."""
    
    image = load_image(image_path)

    # Encode image
    encoder_out = encoder(image)
    encoder_out = encoder_out.view(1, -1, encoder_out.size(-1)).expand(beam_size, -1, -1)

    # Initialize decoder
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * beam_size).to(device)
    seqs = k_prev_words
    top_k_scores = torch.zeros(beam_size, 1).to(device)

    complete_seqs, complete_seqs_scores = [], []

    h, c = decoder.init_hidden_state(encoder_out)
    step = 1

    while step <= 50:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)
        awe, _ = decoder.attention(encoder_out, h)
        gate = decoder.sigmoid(decoder.f_beta(h))
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = F.log_softmax(decoder.fc(h), dim=1)

        scores = top_k_scores.expand_as(scores) + scores
        top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)

        prev_word_inds = top_k_words // len(word_map)
        next_word_inds = top_k_words % len(word_map)
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if complete_inds:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])

        beam_size -= len(complete_inds)
        if beam_size == 0:
            break

        seqs = seqs[incomplete_inds]
        h, c = h[prev_word_inds[incomplete_inds]], c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    best_seq = complete_seqs[i]
    return [rev_word_map[word] for word in best_seq if word not in {'<start>', '<end>', '<pad>'}]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Caption with Beam Search')
    parser.add_argument('--img', '-i', help='Path to image')
    parser.add_argument('--model', '-m', help='Path to model checkpoint')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='Beam size for beam search')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.model, map_location=device)
    decoder = checkpoint['decoder'].to(device).eval()
    encoder = checkpoint['encoder'].to(device).eval()

    # Generate caption
    predicted_caption = caption_image_beam_search(encoder, decoder, args.img, args.beam_size)
    print("Generated Caption:", " ".join(predicted_caption))
