import torch
import torch.nn as nn
from torchvision import transforms, datasets
from model import EncoderCNN, DecoderRNN
import pickle
import os
from torch.autograd import Variable
from Vocabulary import Vocabulary
from PIL import Image
import nltk
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence
from bleu import compute_bleu
batch_size = 20

with open('caption.pkl','rb') as f:
    caption_dict = pickle.load(f)

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
    return images, targets, lengths, image_id

if __name__ == "__main__":

    with open('vocab.pkl','rb') as f:
        vocab = pickle.load(f)
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu' )
    
    img_dataset = ImageFolderWithPaths(root='./test', transform=transform)
    
    dataset_loader = torch.utils.data.DataLoader(img_dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_func)
    
    vocab_size = vocab.index

    cnn = EncoderCNN(512).to(device)
    rnn = DecoderRNN(512, 512, vocab_size).to(device)

    cnn.load_state_dict(torch.load('cnn.pkl'))
    rnn.load_state_dict(torch.load('rnn.pkl'))

    hyp = []
    references = []
    for i, (image, captions, lengths, image_id) in enumerate(dataset_loader):
        image = image.to(device)
        for id in image_id:
            references.append([caption_dict[id].split(' ')[1:]])
        features = cnn.forward(image)
        ids_list = rnn.sample(features)
        ids_list = ids_list.cpu().numpy()
        for ids in ids_list:
            snt = vocab.get_sentence(ids).split()
            hyp.append(snt[1:])
    
    hyp = np.array(hyp)
    references = np.array(references)
    print(hyp.shape,references.shape)
    print(hyp)
    print(compute_bleu(references,hyp))