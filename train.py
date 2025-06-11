import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from modela_new_SAT import Encoder, DecoderWithAttention
from datasets_new_SAT import *
from utils import *  # save_checkpoint, AverageMeter, accuracy etc.
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import json
from torch.optim import lr_scheduler
import os
import torch
from torch.cuda.amp import GradScaler, autocast

# Load word map dynamically
word_map_path = "/kaggle/working/word_map.json"
with open(word_map_path, "r") as j:
    word_map = json.load(j)
vocab_size = len(word_map)

# Device and cudnn setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

# Initialize mixed precision scaler
scaler = GradScaler()

# Hyperparameters (adjust as needed)
alpha_c = 1.0  # attention regularization coefficient
grad_clip = 5.0
print_freq = 100
epochs = 10
batch_size = 32
workers = 4
grad_accum_steps = 2
learning_rate_encoder = 1e-4
learning_rate_decoder = 4e-4

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, grad_accum_steps=2):
    decoder.train()
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)

        with autocast():
            imgs_enc = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs_enc, caps, caplens)

            targets = caps_sorted[:, 1:]

            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            loss = criterion(scores, targets)
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()

        if grad_clip:
            scaler.unscale_(decoder_optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            if encoder_optimizer:
                scaler.unscale_(encoder_optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)

        if (i + 1) % grad_accum_steps == 0:
            scaler.step(decoder_optimizer)
            scaler.update()
            decoder_optimizer.zero_grad()

            if encoder_optimizer:
                scaler.step(encoder_optimizer)
                scaler.update()
                encoder_optimizer.zero_grad()

        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item() * grad_accum_steps, sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        if i % print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]  "
                  f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                  f"Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})  "
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                  f"Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})")


def validate(val_loader, encoder, decoder, criterion):
    decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    references = []
    hypotheses = []

    smoothing = SmoothingFunction().method4

    start = time.time()

    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)

            with autocast():
                if encoder is not None:
                    imgs = encoder(imgs)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

                targets = caps_sorted[:, 1:]

                scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                loss = criterion(scores, targets)
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(accuracy(scores, targets, 5), sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print(f"Validation: [{i}/{len(val_loader)}]  "
                      f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                      f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                      f"Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})")

            allcaps = allcaps[sort_ind]
            # Prepare references (list of lists of tokens)
            for caps_group in allcaps.tolist():
                for caption in caps_group:
                    references.append([w for w in caption if w not in {word_map["<start>"], word_map["<pad>"]}])

            _, preds = torch.max(scores, dim=1)
            preds = preds.tolist()

            for j, length in enumerate(decode_lengths):
                hypotheses.append(preds[j][:length])

    assert len(references) == len(hypotheses)

    bleu4 = corpus_bleu(references, hypotheses, smoothing_function=smoothing)

    print(f"\n * LOSS - {losses.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}, BLEU-4 - {bleu4:.4f}")

    return bleu4


def main():
    # Seed, reproducibility
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)

    # Data transforms (adjust as needed)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Initialize your datasets and dataloaders here
    # Replace DatasetClass with your dataset class from datasets_new_SAT
    train_dataset = DatasetClass(split='train', transform=transform, word_map=word_map)
    val_dataset = DatasetClass(split='val', transform=transform, word_map=word_map)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=workers,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=workers,
                                             collate_fn=val_dataset.collate_fn)

    # Initialize models
    encoder = Encoder().to(device)
    decoder = DecoderWithAttention(vocab_size=vocab_size).to(device)

    # Define loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Define optimizers
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                         lr=learning_rate_encoder)
    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                         lr=learning_rate_decoder)

    best_bleu4 = 0.0
    epochs_since_improvement = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1} of {epochs}")
        train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch,
              grad_accum_steps=grad_accum_steps)
        bleu4 = validate(val_loader, encoder, decoder, criterion)

        # Save checkpoint if best
        is_best = bleu4 > best_bleu4
        best_bleu4 = max(bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
        else:
            epochs_since_improvement = 0

        save_checkpoint('dataset_name', epoch, epochs_since_improvement, encoder, decoder,
                        encoder_optimizer, decoder_optimizer, bleu4, is_best)

        # Early stopping or learning rate adjustment could be added here


if __name__ == "__main__":
    main()
