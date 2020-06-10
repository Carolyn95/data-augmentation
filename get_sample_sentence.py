from rich import print
import numpy as np
import pdb
import pandas as pd


def getSampleSents(input_sents_path, input_labels_path, ioi):
  """Get sample sentences from intent of interests.

  `python get_sample_sentence.py > sample_sentences.txt`

  args:
    input_sents_path: input path of sentences, file format .npy
    input_labels_path: input path of labels, file format .npy
    ioi: intent_of_interest, select from 0, 1, 2
  """
  sents = np.load(input_sents_path, allow_pickle=True)
  labels = np.load(input_labels_path, allow_pickle=True)
  int_labels = np.argmax(labels, axis=1)
  ioi_indexes = [i for i, l in enumerate(int_labels) if l == ioi]

  ioi_sents = sents[ioi_indexes]
  ioi_labels = labels[ioi_indexes]

  for i, s in enumerate(ioi_sents):
    if i != 0 and i % 20 == 0:
      print('===================================\n')
    print(s, '\n\n')


if __name__ == '__main__':
  input_sents_path = 'data/train_sents_mixed.npy'
  input_labels_path = 'data/train_labels_onehot_mixed.npy'
  #{"unknow": 0, "update": 1, "new": 2}
  # 426, 3900, 1295
  getSampleSents(input_sents_path, input_labels_path, 0)
