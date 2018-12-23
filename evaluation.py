# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Evaluation"""

from __future__ import print_function
import os

import sys
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary, deserialize_vocab  # NOQA
import torch
from model import SCAN, xattn_score_t2i1, xattn_score_i2t1, xattn_score_t2i, xattn_score_i2t,\
    cosine_similarity, xattn_score_t2i2, xattn_score_i2t2
from collections import OrderedDict
import time
from torch.autograd import Variable


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):

        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):


    def __init__(self):

        self.meters = OrderedDict()

    def update(self, k, v, n=0):

        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):

        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):

        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):

    batch_time = AverageMeter()
    val_logger = LogCollector()


    model.val_start()

    end = time.time()


    img_embs = None
    cap_embs = None
    cap_lens = None

    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))
    with torch.no_grad():
        for i, (images, captions, lengths, ids) in enumerate(data_loader):

            model.logger = val_logger

            img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths)

            if img_embs is None:
                if img_emb.dim() == 3:
                    img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
                else:
                    img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
                cap_lens = [0] * len(data_loader.dataset)

            img_embs[ids] = img_emb.data.cpu().numpy().copy()
            cap_embs[ids, :max(lengths), :] = cap_emb.data.cpu().numpy().copy()
            for j, nid in enumerate(ids):
                cap_lens[nid] = cap_len[j]


            batch_time.update(time.time() - end)
            end = time.time()

            if i % log_step == 0:
                logging('Test: [{0}/{1}]\t'
                        '{e_log}\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                    i, len(data_loader), batch_time=batch_time,
                    e_log=str(model.logger)))
            del images, captions

    return img_embs, cap_embs, cap_lens


def evalrank(model_path, data_path=None, split='dev', fold5=False):

    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print(opt)
    if data_path is not None:
        opt.data_path = data_path

    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    captions_w = np.load(opt.caption_np + 'caption_np.npy')
    captions_w = torch.from_numpy(captions_w)

    captions_w = captions_w.cuda()


    model = SCAN(opt, captions_w)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs, cap_lens = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] / 5, cap_embs.shape[0]))

    if not fold5:

        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        start = time.time()
        if opt.cross_attn == 't2i':
            sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        elif opt.cross_attn == 'i2t':
            sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        elif opt.cross_attn == 'all':
            sims, label = shard_xattn_all(model, img_embs, cap_embs, cap_lens, opt, shard_size=128)
        else:
            raise NotImplementedError
        end = time.time()
        print("calculate similarity time:", end - start)
        np.save('sim_stage1', sims)

        r, rt = i2t(label, img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ri, rti = t2i(label, img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:

        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            start = time.time()
            if opt.cross_attn == 't2i':
                sims = shard_xattn_t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
            elif opt.cross_attn == 'i2t':
                sims = shard_xattn_i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
            else:
                raise NotImplementedError
            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def softmax(X, axis):

    y = np.atleast_2d(X)
    y = y - np.expand_dims(np.max(y, axis=axis), axis)
    y = np.exp(y)

    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    p = y / ax_sum
    return p


def shard_xattn_t2i(images, captions, caplens, opt, shard_size=128):
    n_im_shard = (len(images) - 1) / shard_size + 1
    n_cap_shard = (len(captions) - 1) / shard_size + 1

    d = np.zeros((len(images), len(captions)))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i, j))
                cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda()
                s = Variable(torch.from_numpy(captions[cap_start:cap_end])).float().cuda()
                l = caplens[cap_start:cap_end]
                sim = xattn_score_t2i1(im, s, l, opt)
                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
        sys.stdout.write('\n')
    return d


def shard_xattn_i2t(images, captions, caplens, opt, shard_size=128):
    n_im_shard = (len(images) - 1) / shard_size + 1
    n_cap_shard = (len(captions) - 1) / shard_size + 1

    d = np.zeros((len(images), len(captions)))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i, j))
                cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda()
                s = Variable(torch.from_numpy(captions[cap_start:cap_end])).float().cuda()
                l = caplens[cap_start:cap_end]
                sim = xattn_score_i2t1(im, s, l, opt)
                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
        sys.stdout.write('\n')
    return d


def shard_xattn_all(model, images, captions, caplens, opt, shard_size=128):
    n_im_shard = (len(images) - 1) / shard_size + 1
    n_cap_shard = (len(captions) - 1) / shard_size + 1
    alpha = 0.8
    d = np.zeros((len(images), len(captions)))
    label = np.zeros((1000, 5000, 500))
    with torch.no_grad():
        for i in range(n_im_shard):
            im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
            for j in range(n_cap_shard):
                sys.stdout.write('\r>> shard_xattn_all batch (%d,%d)' % (i, j))
                cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
                im = Variable(torch.from_numpy(images[im_start:im_end])).float().cuda()
                s = Variable(torch.from_numpy(captions[cap_start:cap_end])).float().cuda()
                l = caplens[cap_start:cap_end]
                sim1, attni = xattn_score_t2i2(im, s, l, opt)
                sim2, attnt = xattn_score_i2t2(im, s, l, opt)
                sims = 0.5 * (sim1 + sim2)
                attnt0 = attnt.size()[0]
                attnt1 = attnt.size()[1]
                attni0 = attni.size()[0]
                attni1 = attni.size()[1]
                attnt = model.FC1(attnt.view(attnt0*attnt1, 1024))
                attni = model.FC1(attni.view(attni0*attni1, 1024))
                attnt = attnt.view(attnt0, attnt1, 500)
                attni = attni.view(attni0, attni1, 500)
                attnt = torch.sigmoid(attnt)
                attni = torch.sigmoid(attni)
                sim_label = cosine_similarity(attni, attnt, 2)
                sim = alpha*sims + (1-alpha)*sim_label
                d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
                label[im_start:im_end, cap_start:cap_end, :] = attni.data.cpu().numpy()
        sys.stdout.write('\n')
    return d, label


def i2t(label, images, captions, caplens, sims, npts=None, return_ranks=False):

    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    indsall = np.zeros((1000, 5))
    im_label = np.zeros((1000 ,500))
    label_idx = np.zeros((1000, 20))
    label_val = np.zeros((1000, 20))

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]
        indsall[index] = inds[0:5]
        labelval = (label[index, inds[0], :]+label[index, inds[1], :]+label[index, inds[2], :])/3.0
        im_label[index, :] = labelval
    np.save('/home/wangzheng/neurltalk/new/SCAN_2loss500_3_0.8_10x/i2t_result.npy', indsall)
    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    label_idx = np.argsort(-im_label, 1)[:, 0:20]
    for k in range(1000):
        label_val[k] = im_label[k][label_idx[k]]
    np.save('/home/wangzheng/neurltalk/new/SCAN_2loss500_3_0.8_10x/label_result2.npy', label_val)
    np.save('/home/wangzheng/neurltalk/new/SCAN_2loss500_3_0.8_10x/label_idx2.npy', label_idx)


    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(label, images, captions, caplens, sims, npts=None, return_ranks=False):

    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    sims = sims.T
    indsall = np.zeros((5000, 3))
    label_val = np.zeros((5000, 20))
    im_label = np.zeros((5000 ,500))
    label_idx = np.zeros((5000, 20))

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]
            indsall[5 * index + i] = inds[0:3]
            labelval = label[inds[0], 5 * index + i, :]
            im_label[5 * index + i, :] = labelval

    np.save('/home/wangzheng/neurltalk/new/SCAN_2loss500_3_0.8_10x/t2i_result.npy', indsall)

    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    label_idx = np.argsort(-im_label, 1)[:, 0:20]
    for k in range(5000):
        label_val[k] = im_label[k][label_idx[k]]
    np.save('/home/wangzheng/neurltalk/new/SCAN_2loss500_3_0.8_10x/label_result_t2i.npy', label_val)
    np.save('/home/wangzheng/neurltalk/new/SCAN_2loss500_3_0.8_10x/label_idx_t2i.npy', label_idx)
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
