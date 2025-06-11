import torch
from torch import nn
import torchvision
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load word_map dynamically
word_map_path = "/kaggle/working/word_map.json"
with open(word_map_path, "r") as j:
    word_map = json.load(j)
vocab_size = len(word_map)

class Encoder(nn.Module):
    """Encoder using ResNet101, outputs feature maps for attention."""

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V1)
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        self.fine_tune()

    def forward(self, images):
        out = self.resnet(images)  # (batch, 2048, H/32, W/32)
        out = self.adaptive_pool(out)  # (batch, 2048, 14, 14)
        out = out.permute(0, 2, 3, 1)  # (batch, 14, 14, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        # Freeze all layers initially
        for p in self.resnet.parameters():
            p.requires_grad = False
        # Unfreeze last two layers for fine tuning
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """Attention Network for computing attention weights."""

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoder output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder hidden state
        self.full_att = nn.Linear(attention_dim, 1)               # linear layer to calculate attention score
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        # decoder_hidden: (batch_size, decoder_dim)
        att1 = self.encoder_att(encoder_out)                      # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)      # (batch_size, 1, attention_dim)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)    # (batch_size, num_pixels)
        alpha = self.softmax(att)                                  # attention weights (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """Decoder with Attention Mechanism for image captioning."""

    def __init__(self, attention_dim=512, embed_dim=300, decoder_dim=512, encoder_dim=2048, vocab_size=None, dropout=0.5):
        super(DecoderWithAttention, self).__init__()

        self.vocab_size = vocab_size if vocab_size is not None else 10000

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(self.vocab_size, embed_dim)           # embedding layer
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)                   # initialize hidden state
        self.init_c = nn.Linear(encoder_dim, decoder_dim)                   # initialize cell state
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)                   # create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, self.vocab_size)                   # final output layer

        self.init_weights()

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        # encoder_out: (batch_size, num_pixels, encoder_dim)
        mean_encoder_out = encoder_out.mean(dim=1)  # average encoding over pixels
        h = self.init_h(mean_encoder_out)            # initialize hidden state
        c = self.init_c(mean_encoder_out)            # initialize cell state
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image features to (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing caption lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embed captions
        embeddings = self.embedding(encoded_captions)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (caption_lengths - 1).tolist()  # decode lengths (minus <end> token)

        # Create tensors to hold predictions and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar
            attention_weighted_encoding = gate * attention_weighted_encoding

            # Decode step
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )

            preds = self.fc(self.dropout_layer(h))  # predict word distribution
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
