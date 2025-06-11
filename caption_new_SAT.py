import torch
import torch.nn.functional as F
import json
import torchvision.transforms as transforms
from PIL import Image
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load word map (index to word)
def load_word_map(word_map_path):
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}
    return word_map, rev_word_map

def load_image(image_path):
    """Load and preprocess the image."""
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0).to(device)

def caption_image_beam_search(encoder, decoder, image_path, word_map, rev_word_map, beam_size=3):
    """Generate caption with beam search."""
    
    image = load_image(image_path)
    encoder_out = encoder(image)  # (1, enc_dim, enc_dim, encoder_dim)
    encoder_out = encoder_out.view(1, -1, encoder_out.size(-1))  # Flatten
    encoder_out = encoder_out.expand(beam_size, *encoder_out.shape[1:])  # beam_size expanded

    k_prev_words = torch.LongTensor([[word_map['<start>']]] * beam_size).to(device)
    seqs = k_prev_words  # (beam_size, 1)
    top_k_scores = torch.zeros(beam_size, 1).to(device)

    complete_seqs = []
    complete_seqs_scores = []

    h, c = decoder.init_hidden_state(encoder_out)
    step = 1

    vocab_size = len(word_map)

    while step <= 50:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (beam_size, embed_dim)
        awe, _ = decoder.attention(encoder_out, h)  # (beam_size, encoder_dim)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))
        scores = F.log_softmax(decoder.fc(h), dim=1)  # (beam_size, vocab_size)

        scores = top_k_scores.expand_as(scores) + scores  # (beam_size, vocab_size)
        top_k_scores, top_k_words = scores.view(-1).topk(beam_size, 0, True, True)  # (beam_size)

        prev_word_inds = top_k_words // vocab_size  # (beam_size)
        next_word_inds = top_k_words % vocab_size   # (beam_size)

        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (beam_size, step+1)

        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if complete_inds:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds].tolist())

        beam_size -= len(complete_inds)
        if beam_size == 0:
            break

        seqs = seqs[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        step += 1

    if len(complete_seqs_scores) == 0:
        complete_seqs = seqs
        complete_seqs_scores = top_k_scores.squeeze(1).tolist()

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    best_seq = complete_seqs[i]

    # Remove special tokens
    caption = [rev_word_map[word] for word in best_seq if word not in {word_map['<start>'], word_map['<end>'], word_map['<pad>']}]
    return caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image caption with beam search")
    parser.add_argument("--img", "-i", required=True, help="Path to image file")
    parser.add_argument("--model", "-m", required=True, help="Path to checkpoint .pth.tar file")
    parser.add_argument("--word_map", "-w", required=True, help="Path to word map JSON file")
    parser.add_argument("--beam_size", "-b", type=int, default=3, help="Beam size for beam search")
    args = parser.parse_args()

    word_map, rev_word_map = load_word_map(args.word_map)

    # Load model checkpoint
    checkpoint = torch.load(args.model, map_location=device)
    decoder = checkpoint['decoder'].to(device).eval()
    encoder = checkpoint['encoder'].to(device).eval()

    caption = caption_image_beam_search(
        encoder, decoder, args.img, word_map, rev_word_map, beam_size=args.beam_size
    )
    print("Generated caption:", " ".join(caption))
