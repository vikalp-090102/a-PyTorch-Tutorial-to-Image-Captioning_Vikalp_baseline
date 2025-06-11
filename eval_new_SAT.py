import os
import json
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import pandas as pd
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm

# Parameters (update these paths as needed)
image_dir = '/kaggle/input/chest-xrays-indiana-university/images/images_normalized'
reports_csv = '/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv'
checkpoint_path = '../BEST_checkpoint.pth.tar'
word_map_file = '/kaggle/working/word_map.json'  # Update to your word_map location

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Load model checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)
decoder = checkpoint['decoder'].to(device).eval()
encoder = checkpoint['encoder'].to(device).eval()

# Load word map
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Load reports dataframe
reports_df = pd.read_csv(reports_csv)

# Image normalization transform
normalize = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def load_image(uid):
    img_path = os.path.join(image_dir, f"{uid}.png")  # Adjust extension if needed
    image = Image.open(img_path).convert("RGB")
    return normalize(image)

def evaluate(beam_size=3, max_caption_length=50):
    references = []
    hypotheses = []

    for idx, row in tqdm(reports_df.iterrows(), total=len(reports_df), desc=f"Evaluating (beam_size={beam_size})"):
        uid = row['uid']
        findings = row['findings']

        # Load and encode image
        image = load_image(uid).unsqueeze(0).to(device)  # (1, 3, 256, 256)
        encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Expand for beam search
        encoder_out = encoder_out.expand(beam_size, num_pixels, encoder_dim)  # (beam_size, num_pixels, encoder_dim)

        # Initialize beam search variables
        k = beam_size
        prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (beam_size, 1)
        sequences = prev_words  # (beam_size, seq_len)
        scores = torch.zeros(k, 1).to(device)  # cumulative log probabilities

        # Initialize LSTM states
        h, c = decoder.init_hidden_state(encoder_out)

        complete_seqs = []
        complete_seqs_scores = []

        step = 1
        while True:
            embeddings = decoder.embedding(prev_words).squeeze(1)  # (k, embed_dim)
            awe, _ = decoder.attention(encoder_out, h)            # (k, encoder_dim)
            gate = decoder.sigmoid(decoder.f_beta(h))              # gating scalar (k, encoder_dim)
            awe = gate * awe

            h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (k, decoder_dim)
            scores_t = F.log_softmax(decoder.fc(h), dim=1)                         # (k, vocab_size)

            # Add previous scores
            scores_expanded = scores + scores_t  # (k, vocab_size)

            # Flatten for topk
            if step == 1:
                # At first step, all beams are the same, so pick top k words
                scores_expanded = scores_expanded[0]  # (vocab_size,)
                top_scores, top_words = scores_expanded.topk(k, 0, True, True)
            else:
                scores_expanded = scores_expanded.view(-1)  # (k*vocab_size,)
                top_scores, top_words = scores_expanded.topk(k, 0, True, True)

            # Extract beam and word indices
            prev_word_inds = top_words // vocab_size  # beam index
            next_word_inds = top_words % vocab_size   # word index

            # Update sequences
            sequences = sequences[prev_word_inds]
            sequences = torch.cat([sequences, next_word_inds.unsqueeze(1)], dim=1)  # append new word

            # Check completed sequences
            incomplete_inds = []
            for i, next_word in enumerate(next_word_inds):
                if next_word == word_map['<end>']:
                    complete_seqs.append(sequences[i].tolist())
                    complete_seqs_scores.append(top_scores[i])
                else:
                    incomplete_inds.append(i)

            # Stop if all sequences complete or max length reached
            if len(incomplete_inds) == 0 or step >= max_caption_length:
                break

            # Prepare for next step
            sequences = sequences[incomplete_inds]
            h = h[prev_word_inds[incomplete_inds]]
            c = c[prev_word_inds[incomplete_inds]]
            encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
            scores = top_scores[incomplete_inds].unsqueeze(1)
            prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            k = len(incomplete_inds)
            step += 1

        # If no complete sequence found, choose best incomplete
        if len(complete_seqs) == 0:
            complete_seqs = sequences.tolist()
            complete_seqs_scores = scores.squeeze(1).tolist()

        # Pick best sequence
        best_seq_idx = complete_seqs_scores.index(max(complete_seqs_scores))
        best_seq = complete_seqs[best_seq_idx]

        # Convert to words, removing start/end/pad tokens
        hypothesis = [w for w in best_seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        hypotheses.append(hypothesis)

        # References (list of list of tokens)
        references.append([findings.lower().split()])

    # Calculate BLEU-4 score
    bleu4 = corpus_bleu(references, hypotheses)
    return bleu4


if __name__ == "__main__":
    beam_size = 3
    print(f"\nBLEU-4 score @ beam size {beam_size}: {evaluate(beam_size):.4f}")
