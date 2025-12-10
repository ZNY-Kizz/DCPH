import numpy as np
import glob
import csv
import operator
import shutil
import torch
import os
import logging
from scipy.spatial.distance import cdist

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def mean_average_precision(database_hash, test_hash, database_labels, test_labels, K = None):  # R = 1000

    query_num = test_hash.shape[0]  # total number for testing
    sim = np.dot(database_hash, test_hash.T)  
    ids = np.argsort(-sim, axis=0)  

    APx = []
    Recall = []

    for i in range(query_num):  # for i=0
        label = test_labels[i, :]  # the first test labels
        if np.sum(label) == 0:  # ignore images with meaningless label in nus wide
            continue
        label[label == 0] = -1
        idx = ids[:, i]
        imatch = np.sum(database_labels[idx[0:K], :] == label, axis=1) > 0
        relevant_num = np.sum(imatch)
        Lx = np.cumsum(imatch)
        Px = Lx.astype(float) / np.arange(1, K + 1, 1)  #

        if relevant_num != 0:
            APx.append(np.sum(Px * imatch) / relevant_num)
        if relevant_num == 0:  # even no relevant image, still need add in APx for calculating the mean
            APx.append(0)

        all_relevant = np.sum(database_labels == label, axis=1) > 0
        all_num = np.sum(all_relevant)
        r = relevant_num / np.float64(all_num)
        Recall.append(r)

    return np.mean(np.array(APx)), np.mean(np.array(Recall)), APx

class CheckpointSaver:
    def __init__(
            self,
            checkpoint_prefix='checkpoint',
            recovery_prefix='recovery',
            checkpoint_dir='',
            recovery_dir='',
            decreasing=False,
            verbose=True,
            max_history=10):

        # state
        self.checkpoint_files = []  # (filename, metric) tuples in order of decreasing betterness
        self.best_epoch = None
        self.best_metric = None
        self.curr_recovery_file = ''
        self.last_recovery_file = ''

        # config
        self.checkpoint_dir = checkpoint_dir
        self.recovery_dir = recovery_dir
        self.save_prefix = checkpoint_prefix
        self.recovery_prefix = recovery_prefix
        self.extension = '.pth.tar'
        self.decreasing = decreasing  # a lower metric is better if True
        self.cmp = operator.lt if decreasing else operator.gt  # True if lhs better than rhs
        self.verbose = verbose
        self.max_history = max_history
        assert self.max_history >= 1

    def save_checkpoint(self, model_list, euclidean_optimizer, hamming_optimizer,cfg, epoch, metric=None):
        assert epoch >= 0
        worst_file = self.checkpoint_files[-1] if self.checkpoint_files else None
        if (len(self.checkpoint_files) < self.max_history
                or metric is None or self.cmp(metric, worst_file[1])):
            if len(self.checkpoint_files) >= self.max_history:
                self._cleanup_checkpoints(1)
            filename = '-'.join([self.save_prefix, str(epoch)]) + self.extension
            save_path = os.path.join(self.checkpoint_dir, filename)
            self._save(save_path, model_list, euclidean_optimizer, hamming_optimizer, cfg, epoch, metric)
            self.checkpoint_files.append((save_path, metric))
            self.checkpoint_files = sorted(
                self.checkpoint_files, key=lambda x: x[1],
                reverse=not self.decreasing)  # sort in descending order if a lower metric is not better

            if self.verbose:
                logging.info("Current checkpoints:")
                for c in self.checkpoint_files[:10]:
                    logging.info(c)

            if metric is not None and (self.best_metric is None or self.cmp(metric, self.best_metric)):
                self.best_epoch = epoch
                self.best_metric = metric
                shutil.copyfile(save_path, os.path.join(self.checkpoint_dir, 'model_best' + self.extension))

        return (None, None) if self.best_metric is None else (self.best_metric, self.best_epoch)

    def _save(self, save_path, model_list, euclidean_optimizer, hamming_optimizer, cfg, epoch, metric=None):
        save_state = {
            'epoch': epoch,
            'state_dict': get_state_dict(model_list),
            'euclidean_optimizer': euclidean_optimizer.state_dict(),
            'hamming_optimizer': hamming_optimizer.state_dict(),
            'cfg': cfg,
        }
        if metric is not None:
            save_state['metric'] = metric
        torch.save(save_state, save_path)

    def _cleanup_checkpoints(self, trim=0):
        trim = min(len(self.checkpoint_files), trim)
        delete_index = self.max_history - trim
        if delete_index <= 0 or len(self.checkpoint_files) <= delete_index:
            return
        to_delete = self.checkpoint_files[delete_index:]
        for d in to_delete:
            try:
                if self.verbose:
                    print('Cleaning checkpoint: ', d)
                os.remove(d[0])
            except Exception as e:
                print('Exception (%s) while deleting checkpoint' % str(e))
        self.checkpoint_files = self.checkpoint_files[:delete_index]

    def save_recovery(self, model, optimizer, args, epoch, model_ema=None, use_amp=False, batch_idx=0):
        assert epoch >= 0
        filename = '-'.join([self.recovery_prefix, str(epoch), str(batch_idx)]) + self.extension
        save_path = os.path.join(self.recovery_dir, filename)
        self._save(save_path, model, optimizer, args, epoch, model_ema, use_amp=use_amp)
        if os.path.exists(self.last_recovery_file):
            try:
                if self.verbose:
                    print('Cleaning recovery', self.last_recovery_file)
                os.remove(self.last_recovery_file)
            except Exception as e:
                print("Exception (%s) while removing %s" % (str(e), self.last_recovery_file))
        self.last_recovery_file = self.curr_recovery_file
        self.curr_recovery_file = save_path

    def find_recovery(self):
        recovery_path = os.path.join(self.recovery_dir, self.recovery_prefix)
        files = glob.glob(recovery_path + '*' + self.extension)
        files = sorted(files)
        if len(files):
            return files[0]
        else:
            return ''
        
def get_state_dict(model_list):
    return [model.module.state_dict() if hasattr(model, 'module') else model.state_dict() for model in model_list]

def HammingDist(q_h,r_h):
    return cdist(q_h, r_h, 'hamming')

def CalRel(q_l,r_l):
    return np.dot(q_l,r_l.T).astype(int)

def average_cumulative_gain(Dist, Rel, K=100):
    n,m = Dist.shape
    if (K < 0) or (K > m):
        K = m
    Gain = Rel
    Rank = np.argsort(Dist)
    
    _ACG = 0
    for g, rnk in zip(Gain, Rank):
        _ACG += g[rnk[:K]].mean()
    return _ACG / n

def normalized_discounted_cumulative_gain(Dist, Rel, K=100):
    n,m = Dist.shape
    if (K < 0) or (K > m):
        K = m
    G = 2 ** Rel - 1
    D = np.log2(2 + np.arange(K))
    Rank = np.argsort(Dist)
    
    _NDCG = 0
    for g, rnk in zip(G, Rank):
        dcg_best = (np.sort(g)[::-1][:K] / D).sum()
        if dcg_best > 0:
            dcg = (g[rnk[:K]] / D).sum()
            _NDCG += dcg / dcg_best
    return _NDCG / n

def weighted_average_precision(Dist, Rel, K=100):
    n,m = Dist.shape
    if (K < 0) or (K > m):
        K = m
    Gain = Rel
    S = (Gain > 0).astype(int)
    pos = np.arange(K) + 1
    Rank = np.argsort(Dist)
    
    _WAP = 0.0
    for s, g, rnk in zip(S, Gain, Rank):
        _rnk = rnk[:K]
        s, g = s[_rnk], g[_rnk]
        n_rel = s.sum()
        if n_rel > 0:
            acg = np.cumsum(g) / pos
            _WAP += (acg * s).sum() / n_rel
    return _WAP / n

    