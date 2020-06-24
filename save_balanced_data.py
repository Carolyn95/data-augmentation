# {'unknow': 0, 'update': 1, 'new': 2}
import numpy as np
import pdb
from collections import Counter
import pandas as pd

# need 520 from train to make it 1500
# update_indexes = [idx for idx, v in enumerate(int_labels) if v == 1]


def getIndexed(labels, count):
  #   count = 520
  needed_idx = []
  for idx, v in enumerate(labels):
    if v != 1:
      needed_idx.append(idx)
    else:
      if count > 0:
        needed_idx.append(idx)
        count -= 1
  return needed_idx


def sliceOriginal(indexes, sents, labels, int_labels):
  sents = sents[indexes]
  labels = labels[indexes]
  int_labels = int_labels[indexes]
  return sents, labels, int_labels


def composeNew(train_sents, train_labels, train_int_labels, valid_sents,
               valid_labels, valid_int_labels):
  mapping = {0: 'unknow', 1: 'update', 2: 'new'}
  # save modelling assets
  all_sents = np.concatenate((train_sents, valid_sents))
  all_labels = np.concatenate((train_labels, valid_labels))
  # np.save('processed_data/all_sents_balanced.npy', all_sents)
  # np.save('processed_data/all_onthot_labels_balanced.npy', all_labels)

  # save human readable csv
  train_str_labels = np.array([mapping[i] for i in train_int_labels])
  valid_str_labels = np.array([mapping[i] for i in valid_int_labels])
  pdb.set_trace()
  all_str_labels = np.concatenate((train_str_labels, valid_str_labels))

  df = pd.DataFrame(columns=['sents', 'intentions'])
  df.sents = all_sents
  df.intentions = all_str_labels
  df.to_csv('processed_data/balanced.csv')


if __name__ == '__main__':
  sents = np.load('processed_data/randomized_sents_1k.npy', allow_pickle=True)
  labels = np.load('processed_data/randomized_labels_1k.npy')

  valid_sents = np.load('data/valid_sents_mixed.npy', allow_pickle=True)
  valid_labels = np.load('data/valid_labels_onehot_mixed.npy')

  int_labels = np.argmax(labels, axis=1)
  int_valid_labels = np.argmax(valid_labels, axis=1)

  indexes = getIndexed(int_labels, 520)
  s_sents, s_labels, s_int_labels = sliceOriginal(indexes, sents, labels,
                                                  int_labels)
  composeNew(s_sents, s_labels, s_int_labels, valid_sents, valid_labels,
             int_valid_labels)
