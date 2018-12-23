# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod


class PrecompDataset1(data.Dataset):


    def __init__(self, data_path, data_split, vocab, vocab_tag):
        self.vocab = vocab
        self.data_split = data_split
        self.vocab_tag = vocab_tag
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)

        self.length = len(self.captions)

        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == 'dev':
            self.length = 1000

    def __getitem__(self, index):
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]

        vocab = self.vocab
        vocab_tag = self.vocab_tag
        captionss = ''
        for i in range(5):
            captionss += self.captions[img_id*5+i]

        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        tokenss = nltk.tokenize.word_tokenize(
            str(captionss).lower().decode('utf-8'))

        caption = []
        caption_tag = [0 for i in range(500)]

        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))

        for token in tokenss:
            if vocab_tag(token) == 0:
                continue
            else:
                caption_tag[vocab_tag(token)-1] = 1


        target = torch.Tensor(caption)
        target_tag = torch.Tensor(caption_tag).view(1, 500)

        return image, target, target_tag, index, img_id



    def __len__(self):
        return self.length


class PrecompDataset2(data.Dataset):


    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)

        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):

        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length



def collate_fn1(data):


    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, captions_tag, ids, img_ids = zip(*data)


    images = torch.stack(images, 0)


    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, captions_tag, lengths, ids

def collate_fn2(data):


    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)


    images = torch.stack(images, 0)


    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids

def get_precomp_loader1(data_path, data_split, vocab, vocab_tag, opt, batch_size=100,
                       shuffle=True, num_workers=2):

    dset = PrecompDataset1(data_path, data_split, vocab, vocab_tag)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn1)
    return data_loader

def get_precomp_loader2(data_path, data_split, vocab, opt, batch_size=100,
                       shuffle=True, num_workers=2):

    dset = PrecompDataset2(data_path, data_split, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              collate_fn=collate_fn2)
    return data_loader

def get_loaders(data_name, vocab, vocab_tag, batch_size, workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader1(dpath, 'train', vocab, vocab_tag, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader2(dpath, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, data_name, vocab, batch_size,
                    workers, opt):
    dpath = os.path.join(opt.data_path, data_name)
    test_loader = get_precomp_loader2(dpath, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader
