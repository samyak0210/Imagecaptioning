import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.vgg = models.vgg11(pretrained=True)
        in_features = self.vgg.classifier[6].in_features
        self.linear = nn.Linear(in_features, embed_size)
        self.vgg.classifier[6] = self.linear
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, image):
        features = self.vgg(image)
        return features

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, caption, lengths):
        embeds = self.embed(caption)
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True)
        lstm_out, _ =  self.lstm(packed)
        out = self.linear(lstm_out[0])
        return out

    def sample(self, features, max_word_length=20):
        hidden = None
        caption = []
        inputs = features.unsqueeze(1)
        for i in range(max_word_length):
            lstm_out, hidden = self.lstm(inputs, hidden)
            linear_out = self.linear(lstm_out.squeeze(1))
            word = linear_out.max(dim=1)[1]
            caption.append(word)
            inputs = self.embed(word)
            inputs = inputs.unsqueeze(1)
        caption = torch.stack(caption, 1)
        return caption