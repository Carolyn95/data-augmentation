"""
wrap pairwise as a easily plug and play component

"""

# onehot and save onehot labels

# if train & valid => read data separately
# else split data, stractified sampling

# random sample method

# labels interested

from pathlib import Path
import pdb
import numpy as np
import pickle as pkl
from rich import print


class DataReader():

  def __init__(self, save_path, labels, sents, **kwargs):
    for k, v in kwargs.items():
      if 'label' in k:
        self.labels = np.concatenate((v, labels))
      elif 'sent' in k:
        self.sents = np.concatenate((v, sents))
      else:
        raise ValueError('Expected valid_labels and valid_sents as pair input')
    self.labels = labels
    self.sents = sents
    self.save_dir = save_path

  def takeTwoLabels(self, loi):
    self.loi = loi
    # loi: label of interest
    indexes = [i for i, l in enumerate(self.labels) if l in loi]
    self.labels = self.labels[indexes]
    self.sents = self.sents[indexes]
    print('labels length is {}, sentences length is {}'.format(
        len(self.labels), len(self.sents)))

  def shuffleData(self):
    shuffler = np.random.permutation(len(self.labels))
    print(shuffler)
    self.labels = self.labels[shuffler]
    self.sents = self.sents[shuffler]

  def sampleData(self, sampling_num=None):
    cats, counts = np.unique(self.labels, return_counts=True)
    cat_count = {cat: count for cat, count in zip(cats, counts)}
    print(cat_count)
    with open(self.save_dir / 'cat_count.pkl', 'wb') as f:
      pkl.dump(cat_count, f)
    sampling_num = sampling_num if sampling_num else min(counts)
    indexes = {}
    for cat in cats:
      temp_indexes = np.array(
          [i for i, l in enumerate(self.labels) if l == cat][:sampling_num])
      indexes[cat] = temp_indexes

    self.labels = np.concatenate([self.labels[indexes[c]] for c in cats])
    self.sents = np.concatenate([self.sents[indexes[c]] for c in cats])
    file_name = '_'.join(self.loi)
    np.save(self.save_dir / (file_name + '_labels.npy'), self.labels)
    np.save(self.save_dir / (file_name + '_sents.npy'), self.sents)

  def splitData(self, ratio=.2):
    cutoff = int(len(self.labels) * ratio)
    self.train_labels = self.labels[cutoff:]
    self.train_sents = self.sents[cutoff:]
    self.valid_labels = self.labels[:cutoff]
    self.valid_sents = self.sents[:cutoff]
    file_name = '_'.join(self.loi)
    np.save(self.save_dir / (file_name + '_labels_train.npy'),
            self.train_labels)
    np.save(self.save_dir / (file_name + '_sents_train.npy'), self.train_sents)
    np.save(self.save_dir / (file_name + '_labels_valid.npy'),
            self.valid_labels)
    np.save(self.save_dir / (file_name + '_sents_valid.npy'), self.valid_sents)

    print('train labels length is {}, \
          train sents length is {}, \
          valid labels length is {} \
          valid sents length is {}'.format(len(self.train_labels),
                                           len(self.train_sents),
                                           len(self.valid_labels),
                                           len(self.valid_sents)))


if __name__ == '__main__':
  save_dir = 'pairwise_exprmt/data_v1'
  seril_no = '14-4'  # 13-0 | 13-1 | 13-2 | 13-3 | 13-4
  save_path = save_dir + '/' + seril_no + '/'
  save_path = Path(save_path)
  save_path.mkdir(parents=True, exist_ok=True)

  labels = np.load('data_v1/all_feed_labels.npy', allow_pickle=True)
  sents = np.load('data_v1/all_feed_emails.npy', allow_pickle=True)
  # valid_labels = np.load('data/valid_labels_str_mixed.npy', allow_pickle=True)
  # valid_sents = np.load('data/valid_sents_mixed.npy', allow_pickle=True)
  dr = DataReader(save_path, labels, sents)
  dr.takeTwoLabels(['update', 'new'])
  dr.shuffleData()
  dr.sampleData()
  dr.shuffleData()
  dr.splitData()
