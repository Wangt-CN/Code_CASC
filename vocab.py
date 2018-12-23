
import nltk
from collections import Counter
import argparse
import os
import json

annotations = {
    'coco_precomp': ['train_caps.txt', 'dev_caps.txt'],
    'f30k_precomp': ['train_caps.txt', 'dev_caps.txt'],
}


class Vocabulary(object):

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def serialize_vocab(vocab, dest):
    d = {}
    d['word2idx'] = vocab.word2idx
    d['idx2word'] = vocab.idx2word
    d['idx'] = vocab.idx
    with open(dest, "w") as f:
        json.dump(d, f)


def deserialize_vocab(src):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab


def from_txt(txt):
    captions = []
    with open(txt, 'rb') as f:
        for line in f:
            captions.append(line.strip())
    return captions


def build_vocab(data_path, data_name, caption_file, threshold):

    stopword_list = list(set(nltk.corpus.stopwords.words('english')))
    counter = Counter()
    counter_tag = Counter()
    for path in caption_file[data_name]:
        full_path = os.path.join(os.path.join(data_path, data_name), path)
        captions = from_txt(full_path)
        tag_list = ['NN', 'NNS', 'NNP', 'NNPS']
        for i, caption in enumerate(captions):
            tokens = nltk.tokenize.word_tokenize(
                caption.lower().decode('utf-8'))
            tokens_tag = nltk.pos_tag(tokens)
            tokens_tag = [j[0] for j in tokens_tag if j[1] in tag_list]
            tokens_tag = [k for k in tokens_tag if k not in stopword_list]
            counter.update(tokens)
            counter_tag.update(tokens_tag)

            if i % 1000 == 0:
                print("[%d/%d] tokenized the captions." % (i, len(captions)))

    # Discard if the occurrence of the word is less than min_word_cnt.
    words_tag = sorted(counter_tag.items(), key=lambda x:x[1], reverse=True)[0:500]
    words_tag = [word[0] for word in words_tag]
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    vocab_tag = Vocabulary()
    vocab_tag.add_word('<unk>')

    # Add words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)

    for i, word in enumerate(words_tag):
        vocab_tag.add_word(word)

    return vocab, vocab_tag


def main(data_path, data_name):
    vocab, vocab_tag = build_vocab(data_path, data_name, caption_file=annotations, threshold=4)
    serialize_vocab(vocab, 'vocab/%s_vocab.json' % data_name)
    serialize_vocab(vocab_tag, 'vocab/%s_vocab_tag.json' % data_name)
    print("Saved vocabulary file to ", 'vocab/%s_vocab.json, %s_vocab_tag.json' %(data_name, data_name))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--data_name', default='f30k_precomp',
                        help='{coco,f30k}_precomp')
    opt = parser.parse_args()
    main(opt.data_path, opt.data_name)
