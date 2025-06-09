import torch
from torch import nn
import torchvision
import json

# Set device: GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load word_map for dynamic vocabulary sizing
word_map_path = "/kaggle/working/word_map.json"
with open(word_map_path, "r") as j:
    word_map = json.load(j)
vocab_size = len(word_map)  # Dynamically set vocab size

class Encoder(nn.Module):
    """Encoder tailored for Indiana chest X-ray dataset."""
    
    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size
        
        resnet = torchvision.models.resnet101(pretrained=True)
        # Remove final classification layers
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # Resize feature maps to a fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))
        # Allow fine-tuning of deeper layers
        self.fine_tune()
    
    def forward(self, images):
        """Forward propagation."""
        out = self.resnet(images)  # (batch_size, 2048, H/32, W/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, 14, 14)
        out = out.permute(0, 2, 3, 1)  # (batch_size, 14, 14, 2048)
        return out
    
    def fine_tune(self, fine_tune=True):
        """Allow fine-tuning of ResNet layers beyond block 4."""
        for p in self.resnet.parameters():
            p.requires_grad = False
        # Unfreeze layers from the 5th block onward if fine_tuning is enabled
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune

class Attention(nn.Module):
    """Attention Network tailored for medical image captioning."""
    
    def __init__(self, encoder_dim=2048, decoder_dim=512, attention_dim=512):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Transform encoder's output
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)              # Compute attention scores 
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)                           # Normalize scores
        
    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.
        :param encoder_out: encoded image feature vectors, shape: [batch, num_pixels, encoder_dim]
        :param decoder_hidden: previous decoder hidden state, shape: [batch, decoder_dim]
        :return: attention weighted encoding, alpha (attention weights)
        """
        att1 = self.encoder_att(encoder_out)                     # [batch, num_pixels, attention_dim]
        att2 = self.decoder_att(decoder_hidden)                  # [batch, attention_dim]
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # [batch, num_pixels]
        alpha = self.softmax(att)                                # [batch, num_pixels]
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # [batch, encoder_dim]
        return attention_weighted_encoding, alpha

class DecoderWithAttention(nn.Module):
    """Decoder with Attention mechanism."""
    
    def __init__(self, attention_dim=512, embed_dim=300, decoder_dim=512, encoder_dim=2048, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout
        
        # Initialize the attention network
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)      # Embedding layer
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # LSTMCell
        
        self.init_h = nn.Linear(encoder_dim, decoder_dim)         # Initialize hidden state
        self.init_c = nn.Linear(encoder_dim, decoder_dim)         # Initialize cell state
        
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)         # Linear layer to create a gating scalar
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)              # Final output layer for predictions
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize layers for better convergence."""
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
    
    def init_hidden_state(self, encoder_out):
        """Create the initial hidden and cell states given the encoder output."""
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c
    
    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward pass for decoding.
        :param encoder_out: encoded images, shape: [batch, enc_image_size, enc_image_size, encoder_dim]
        :param encoded_captions: encoded captions, shape: [batch, max_caption_length]
        :param caption_lengths: caption lengths, shape: [batch, 1]
        :return: predictions, encoded_captions, decode_lengths, alphas, sort_ind
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        
        # Flatten image embeddings
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # [batch, num_pixels, encoder_dim]
        num_pixels = encoder_out.size(1)
        
        # Sort data by decreasing lengths
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        
        # Embeddings for decoded captions
        embeddings = self.embedding(encoded_captions)  # [batch, max_caption_length, embed_dim]
        
        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)
        
        # Determine decode lengths
        decode_lengths = (caption_lengths - 1).tolist()
        
        # Create tensors to hold word prediction scores and attention weights
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)
        
        # Decode each time step
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            # Get attention weighted encoding
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            
            # Gate the attention weighted encoding
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # Update LSTM hidden state
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            
            preds = self.fc(self.dropout_layer(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        
        return predictions, encoded_captions, decode_lengths, alphas, sort_ind
