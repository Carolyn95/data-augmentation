# save train according to labels
# use permutation to get random numbers
# get 520 updates from train randomly
# use best typology each folder saves one result

import pdb
import numpy as np
from pathlib import Path
from collections import Counter
# {0: 'unknow', 1: 'update', 2: 'new'}


class DataGenerator():

  def __init__(self):
    self.sents = np.load('processed_data/randomized_sents_1k.npy',
                         allow_pickle=True)
    self.labels = np.load('processed_data/randomized_labels_1k.npy')
    self.mapping = {0: 'unknow', 1: 'update', 2: 'new'}

  def getValidationData(self):
    self.valid_sents = np.load('data/valid_sents_mixed.npy', allow_pickle=True)
    self.valid_labels = np.load('data/valid_labels_onehot_mixed.npy')

  def composeData(self):
    self.all_sents = np.concatenate((self.sents, self.valid_sents))
    self.all_labels = np.concatenate((self.labels, self.valid_labels))
    self.all_int_labels = np.argmax(self.all_labels, axis=1)
    shuffler = np.random.permutation(len(self.all_int_labels))
    print(shuffler)
    self.all_sents = self.all_sents[shuffler]
    self.all_labels = self.all_labels[shuffler]
    self.all_int_labels = self.all_int_labels[shuffler]

  def getIndexes(self, count=1500):
    self.needed_idx = []
    for idx, v in enumerate(self.all_int_labels):
      if v != 1:
        self.needed_idx.append(idx)
      else:
        if count > 0:
          self.needed_idx.append(idx)
          count -= 1

  def saveData(self, path=None, split_rate=0.2):
    if path is None:
      raise ValueError
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    self.sel_sents = self.all_sents[self.needed_idx]
    self.sel_labels = self.all_labels[self.needed_idx]
    self.sel_int_labels = self.all_int_labels[self.needed_idx]
    np.save(path / 'sents.npy', self.sel_sents)
    np.save(path / 'labels.npy', self.sel_labels)
    np.save(path / 'int_labels.npy', self.sel_int_labels)
    split_ = int(split_rate * len(self.sel_labels))
    self.train_sents = self.sel_sents[split_:]
    self.valid_sents = self.sel_sents[:split_]
    self.train_labels = self.sel_labels[split_:]
    self.valid_labels = self.sel_labels[:split_]
    self.train_int_labels = self.sel_int_labels[split_:]
    self.valid_int_labels = self.sel_int_labels[:split_]
    np.save(path / 'train_sents.npy', self.train_sents)
    np.save(path / 'valid_sents.npy', self.valid_sents)
    np.save(path / 'train_labels.npy', self.train_labels)
    np.save(path / 'valid_labels.npy', self.valid_labels)
    np.save(path / 'train_int_labels.npy', self.train_int_labels)
    np.save(path / 'valid_int_labels.npy', self.valid_int_labels)

  def saveCSV(self, path=None):
    if path is None:
      raise ValueError
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(columns=['sents', 'intentions'])
    df.sents = self.sel_sents
    df.intentions = self.sel_labels
    df.to_csv(path / 'all.csv')


if __name__ == '__main__':
  dg = DataGenerator()
  dg.getValidationData()
  count = 0
  while count < 5:
    dg.composeData()
    dg.getIndexes()
    path = '20200624_exprmt_' + str(count) + '/'
    dg.saveData(path)
    count += 1
