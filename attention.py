import torch
from torch import nn

class Attention(nn.Module):
    """
    Attention Network tailored for medical image captioning.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: Feature size of encoded images (2048 for ResNet-101)
        :param decoder_dim: Size of decoder's LSTM hidden state
        :param attention_dim: Size of attention layer
        """
        super(Attention, self).__init__()
        
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # Image feature transformation
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # Decoder hidden state transformation
        self.full_att = nn.Linear(attention_dim, 1)  # Attention score layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # Normalize attention weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: Encoded image tensor (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: Previous decoder hidden state (batch_size, decoder_dim)
        :return: Attention-weighted encoding, attention weights
        """
        # Compute attention scores
        att1 = self.encoder_att(encoder_out)  # Transform image features
        att2 = self.decoder_att(decoder_hidden)  # Transform hidden state
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # Apply activation & score layer
        
        # Compute attention weights
        alpha = self.softmax(att)  # Normalize scores (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # Weighted sum (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha
