import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from modela_new_SAT import Encoder, DecoderWithAttention
from datasets_new_SAT import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import json
from torch.optim import lr_scheduler

# Data parameters
data_folder = "/kaggle/input/chest-xrays-indiana-university"  # Update to Indiana dataset path
data_name = "indiana_chest_xray"  # Modify to match dataset structure

# Load word map dynamically
word_map_path = "/kaggle/working/word_map.json"
with open(word_map_path, "r") as j:
    word_map = json.load(j)
vocab_size = len(word_map)

# Model parameters
emb_dim = 512  # Word embedding size
attention_dim = 512  # Attention layer size
decoder_dim = 512  # Decoder RNN size
dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # Enable optimization for fixed input sizes

# Training parameters
start_epoch = 0
epochs = 50  # Reduce epochs since medical datasets typically converge faster
epochs_since_improvement = 0  # Track BLEU improvements
batch_size = 16  # Reduce for X-ray images (depends on GPU)
workers = 4  # Increase for faster data loading
encoder_lr = 5e-5  # Adjust learning rate for fine-tuning on X-ray images
decoder_lr = 3e-4  # Slightly lower learning rate for stable training
grad_clip = 5.0  # Gradient clipping for stability
alpha_c = 1.0  # Attention regularization
best_bleu4 = 0.0  # Initialize BLEU-4 score
print_freq = 50  # More frequent logging for medical training
fine_tune_encoder = True  # Enable fine-tuning since X-ray features may differ
checkpoint = "/kaggle/working/checkpoint.pth"  # Save model in Kaggle working directory

def main():
    
    """Training and validation."""

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder, data_name, word_map

    # Initialize / load checkpoint
    if checkpoint is None:
        decoder = DecoderWithAttention(attention_dim=attention_dim,
                                       embed_dim=emb_dim,
                                       decoder_dim=decoder_dim,
                                       vocab_size=vocab_size,
                                       dropout=dropout)

        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=decoder_lr)

        encoder = Encoder()
        encoder.fine_tune(fine_tune_encoder)

        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=encoder_lr) if fine_tune_encoder else None

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        epochs_since_improvement = checkpoint["epochs_since_improvement"]
        best_bleu4 = checkpoint["bleu-4"]
        decoder = checkpoint["decoder"]
        decoder_optimizer = checkpoint["decoder_optimizer"]
        encoder = checkpoint["encoder"]
        encoder_optimizer = checkpoint["encoder_optimizer"]

        if fine_tune_encoder and encoder_optimizer is None:
            encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=encoder_lr)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    normalize = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        IndianaXrayDataset(image_folder, reports_csv, split="TRAIN", transform=normalize),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        IndianaXrayDataset(image_folder, reports_csv, split="VAL", transform=normalize),
        batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )

    print("Training initialized...")


    # Epochs
    import torch.optim.lr_scheduler as lr_scheduler

    decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr=decoder_lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(decoder_optimizer, mode="max", factor=0.8, patience=8)

    for epoch in range(start_epoch, epochs):
    # Terminate training if BLEU score has stagnated for too long
        if epochs_since_improvement >= 20:
            print("\nTraining stopped due to lack of improvement.")
            break

    # One epoch's training
    train(train_loader=train_loader,
          encoder=encoder,
          decoder=decoder,
          criterion=criterion,
          encoder_optimizer=encoder_optimizer,
          decoder_optimizer=decoder_optimizer,
          epoch=epoch)

    # One epoch's validation
    recent_bleu4 = validate(val_loader=val_loader,
                            encoder=encoder,
                            decoder=decoder,
                            criterion=criterion)

    # Check for improvement
    is_best = recent_bleu4 > best_bleu4
    best_bleu4 = max(recent_bleu4, best_bleu4)

    if not is_best:
        epochs_since_improvement += 1
        print(f"\nEpochs since last improvement: {epochs_since_improvement}")
        scheduler.step(recent_bleu4)  # Adjust LR based on BLEU score
    else:
        epochs_since_improvement = 0

    # Save checkpoint only if improvement occurs
    if is_best:
        save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


import torch
import time
from torch.nn.utils.rnn import pack_padded_sequence
from torch.cuda.amp import GradScaler, autocast  # Mixed precision for efficiency

scaler = GradScaler()  # Helps with automatic mixed precision training

def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, grad_accum_steps=2):
    """
    Performs one epoch's training with gradient accumulation for X-ray images.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    :param grad_accum_steps: number of batches to accumulate gradients before updating weights
    """

    decoder.train()  # Enable training mode (dropout active)
    encoder.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    # Batches
    for i, (imgs, caps, caplens) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU
        imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)

        with autocast():  # Mixed precision training
            # Forward pass
            imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

            targets = caps_sorted[:, 1:]  # Shift to exclude <start>

            # Pack padded sequence for loss calculation
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Loss function (with label smoothing to handle rare classes)
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Backpropagation with gradient accumulation
        loss /= grad_accum_steps  # Normalize loss
        scaler.scale(loss).backward()

        # Clip gradients
        if grad_clip:
            scaler.unscale_(decoder_optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
            if encoder_optimizer:
                scaler.unscale_(encoder_optimizer)
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), grad_clip)

        # Update weights after `grad_accum_steps` batches
        if (i + 1) % grad_accum_steps == 0:
            scaler.step(decoder_optimizer)
            scaler.update()

            if encoder_optimizer:
                scaler.step(encoder_optimizer)
                scaler.update()

            decoder_optimizer.zero_grad()
            if encoder_optimizer:
                encoder_optimizer.zero_grad()

        # Track accuracy & loss
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item() * grad_accum_steps, sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]  "
                  f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                  f"Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})  "
                  f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                  f"Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})")

import torch
import time
from torch.nn.utils.rnn import pack_padded_sequence
from torch.cuda.amp import autocast  # Mixed precision for validation

def validate(val_loader, encoder, decoder, criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss function
    :return: BLEU-4 score
    """
    decoder.eval()  # Switch to evaluation mode (no dropout)
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()

    references = []
    hypotheses = []

    # Disable gradient calculations to save memory
    with torch.no_grad():
        for i, (imgs, caps, caplens, allcaps) in enumerate(val_loader):
            imgs, caps, caplens = imgs.to(device), caps.to(device), caplens.to(device)

            # Forward pass with mixed precision
            with autocast():
                if encoder is not None:
                    imgs = encoder(imgs)
                scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(imgs, caps, caplens)

                targets = caps_sorted[:, 1:]  # Shift captions to remove <start>

                scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                loss = criterion(scores, targets)
                loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(accuracy(scores, targets, 5), sum(decode_lengths))
            batch_time.update(time.time() - start)
            start = time.time()

            if i % print_freq == 0:
                print(f"Validation: [{i}/{len(val_loader)}]  "
                      f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                      f"Loss {losses.val:.4f} ({losses.avg:.4f})  "
                      f"Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})")

            # Store references (true captions) and hypotheses (predictions)
            allcaps = allcaps[sort_ind]  # Sort captions in alignment with sorted images
            references.extend([
                [w for w in caption if w not in {word_map["<start>"], word_map["<pad>"]}]
                for captions in allcaps.tolist()
                for caption in captions
            ])

            _, preds = torch.max(scores, dim=2)  # Get predicted words
            preds = preds.tolist()

            hypotheses.extend([
                preds[j][:decode_lengths[j]]
                for j in range(len(preds))
            ])

        assert len(references) == len(hypotheses)

        # Calculate BLEU-4 score with smoothing to avoid floating-point instability
        bleu4 = corpus_bleu(references, hypotheses, smoothing_function=lambda x: 1.0)

        print(f"\n * LOSS - {losses.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}, BLEU-4 - {bleu4:.4f}")

    return bleu4

if __name__ == "__main__":
    main()
