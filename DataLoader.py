import nltk
import torch
import os, pickle
from PIL import Image
from torchvision import datasets

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
    data.sort(key=lambda x:len(nltk.tokenize.word_tokenize(x[3])), reverse=True)
    images, _, image_id, _ = zip(*data)
    caption = [caption2ids(caption_dict[i]) for i in image_id]

    images = torch.stack(images, 0)

    lengths = [len(cap) for cap in caption]
    targets = torch.zeros(len(caption), max(lengths)).long()

    for i, cap in enumerate(caption):
        end = lengths[i]
        targets[i, :end] = torch.LongTensor(cap[:end])
    return images, targets, lengths
