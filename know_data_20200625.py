# know how categories distributing in training and testing
# print out some samples in each category
# function takes in a folder path
# according to label, print out some samples
# {0: 'unknow', 1: 'update', 2: 'new'}
import pdb
import numpy as np
import numpy as np
from rich import print


def diveInData(path):
  train_sents = np.load(path + 'train_sents.npy', allow_pickle=True)
  train_labels = np.load(path + 'train_int_labels.npy')
  test_sents = np.load(path + 'valid_sents.npy', allow_pickle=True)
  test_labels = np.load(path + 'valid_int_labels.npy')

  def aggregateStats(label_array):
    uniq_elements, cnt_elements = np.unique(label_array, return_counts=True)
    stats = [{e: cnt} for e, cnt in zip(uniq_elements, cnt_elements)]
    print(stats)
    return uniq_elements

  train_cats = aggregateStats(train_labels)
  test_cats = aggregateStats(test_labels)

  def printData(data_array, label_array, cats, cap=20):
    for cat in cats:
      sub_labels = [i for i, l in enumerate(label_array) if l == cat]
      sub_sents = train_sents[sub_labels]
      np.random.shuffle(sub_sents)
      count = 0
      while count < cap:
        print("{} - {}: {}".format(cat, count, sub_sents[count]))
        count += 1
      print('----------')

  print('===== Train =====')
  printData(train_sents, train_labels, train_cats)
  print('===== Test =====')
  printData(test_sents, test_labels, test_cats)


if __name__ == '__main__':
  path = '20200624_exprmt_0/'
  diveInData(path)
