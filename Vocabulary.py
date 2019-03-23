import nltk
from collections import Counter
from caption import load_captions
import pickle

class Vocabulary():
    def __init__(self, captions_dict, threshold):
        self.word2id = {}
        self.id2word = {}
        self.index = 0
        self.build(captions_dict, threshold)

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.index
            self.id2word[self.index] = word
            self.index += 1

    def get_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        return self.word2id['<unk>']

    def get_word(self, index):
        return self.id2word[index]

    def build(self, caption_dict, threshold):
        tokens = []
        for _, caption in caption_dict.items():
            tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))
        counter = Counter(tokens)

        words = [word for word, cnt in counter.items() if cnt>=threshold]
        self.add_word('<unk>')
        self.add_word('<start>')
        self.add_word('<end>')
        self.add_word('<pad>')
        for word in words:
            self.add_word(word)

    def get_sentence(self, ids_list):
        sent = ''
        for cur_id in ids_list:
            cur_word = self.id2word[cur_id.item()]
            if cur_word == '<end>':
                break
            sent += cur_word + ' '
        return sent

if __name__ == '__main__':
    captions_dict = load_captions()
    vocab = Vocabulary(captions_dict, 5)
    with open('vocab.pkl','wb') as f:
        pickle.dump(vocab, f)