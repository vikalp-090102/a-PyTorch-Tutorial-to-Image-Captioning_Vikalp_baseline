import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import pandas as pd
import os
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import torch.nn.functional as F

# Parameters
image_dir = '/kaggle/input/chest-xrays-indiana-university/images/images_normalized'  # Update this path
reports_csv = '/kaggle/input/chest-xrays-indiana-university/indiana_reports.csv'  # Update path
checkpoint_path = '../BEST_checkpoint.pth.tar'
word_map_file = '/path/to/word_map.json'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Load model
checkpoint = torch.load(checkpoint_path)
decoder = checkpoint['decoder'].to(device).eval()
encoder = checkpoint['encoder'].to(device).eval()

# Load word map
with open(word_map_file, 'r') as j:
    word_map = json.load(j)
rev_word_map = {v: k for k, v in word_map.items()}
vocab_size = len(word_map)

# Load reports data
reports_df = pd.read_csv(reports_csv)

# Normalization transform
normalize = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(uid):
    """Load an image given its UID."""
    img_path = os.path.join(image_dir, f"{uid}.png")  # Adjust extension as needed
    image = Image.open(img_path).convert("RGB")
    return normalize(image)

def evaluate(beam_size=1):
    """Generate captions and evaluate BLEU-4."""
    references = []
    hypotheses = []

    for idx, row in tqdm(reports_df.iterrows(), desc=f"EVALUATING @ BEAM SIZE {beam_size}"):
        uid = row['uid']
        findings = row['findings']

        # Load image
        image = load_image(uid).unsqueeze(0).to(device)  # (1, 3, 256, 256)

        # Encode image
        encoder_out = encoder(image)
        encoder_out = encoder_out.view(1, -1, encoder_out.size(-1))
        encoder_out = encoder_out.expand(beam_size, *encoder_out.shape[1:])

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

            prev_word_inds = top_k_words // vocab_size
            next_word_inds = top_k_words % vocab_size
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

        best_seq = complete_seqs[complete_seqs_scores.index(max(complete_seqs_scores))]
        hypothesis = [w for w in best_seq if w not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
        hypotheses.append(hypothesis)
        references.append([findings.split()])

    bleu4 = corpus_bleu(references, hypotheses)
    return bleu4

if __name__ == "__main__":
    beam_size = 3
    print(f"\nBLEU-4 score @ beam size {beam_size}: {evaluate(beam_size):.4f}")
