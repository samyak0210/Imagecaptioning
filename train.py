import torch
import torch.nn as nn
import pickle
import os, nltk
from torch.autograd import Variable
from torchvision import transforms, datasets
from model import EncoderCNN, DecoderRNN
from Vocabulary import Vocabulary
from caption import load_captions
from torch.nn.utils.rnn import pack_padded_sequence
import time
import numpy as np

global caption_dict
global vocab

with open('caption.pkl','rb') as f:
    caption_dict = pickle.load(f)

with open('vocab.pkl','rb') as f:
    vocab = pickle.load(f)
    
class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        name = path.split('/')[-1]
        caption = caption_dict[name]
        tuple_with_path = (original_tuple + (name, caption))
        return tuple_with_path


def caption2ids(caption):
    if caption is None:
        tokens = ['<unk>']
    else:
        tokens = nltk.tokenize.word_tokenize(caption.lower())
    vec = []
    vec.append(vocab.get_id('<start>'))
    vec.extend([vocab.get_id(word) for word in tokens])
    vec.append(vocab.get_id('<end>'))
    return vec

def collate_func(data):
    data.sort(key=lambda x:len(caption2ids(x[3])), reverse=True)
    images, _, image_id, _ = zip(*data)
    caption = [caption2ids(caption_dict[i]) for i in image_id]

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in caption]
    targets = torch.zeros(len(caption), max(lengths)).long()

    for i, cap in enumerate(caption):
        end = lengths[i]
        targets[i, :end] = torch.LongTensor(cap[:end])
    return images, targets, lengths


transform = transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 20
num_epochs = 100
img_dataset = ImageFolderWithPaths(root='./train', transform=transform)
dataset_loader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_func)

vocab_size = vocab.index

cnn = EncoderCNN(512).to(device)
rnn = DecoderRNN(512, 512, vocab_size).to(device)

criterion = nn.CrossEntropyLoss()
params = list(cnn.linear.parameters()) + list(rnn.parameters())
optimizer = torch.optim.Adam(params, lr=1e-3)


for epoch in range(num_epochs):
    tic = time.time()

    for i, (image, captions, lengths) in enumerate(dataset_loader):

        image = image.to(device)
        captions = captions.to(device)
        targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

        cnn.zero_grad()
        rnn.zero_grad()

        cnn_out = cnn.forward(image)
        lstm_out = rnn.forward(cnn_out, captions, lengths)
        loss = criterion(lstm_out, targets)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'
                .format(epoch, num_epochs, i, len(dataset_loader), loss.item(), np.exp(loss.item()))) 

    toc = time.time()
    print('epoch %d time %.2f mins'
          % (epoch, (toc - tic) / 60))

torch.save(cnn.state_dict(), 'cnn.pkl')
torch.save(rnn.state_dict(), 'rnn.pkl')
