import numpy as np
import pdb
import pickle as pkl
import re
import os
import operator
from pathlib import Path
from rich import print
from collections import Counter


class BaseArray():

  def __init__(self, data):
    self.data = data
    print('# of original data: {} '.format(len(self.data)))
    self.adict = {
        'unknown': 'unknown',
        'update': 'update',
        'new': 'new',
        'resolved': 'resolved'
    }

  def replace(self, text):
    regex = re.compile("|".join(map(re.escape, self.adict.keys())))
    return ';'.join(regex.findall(text))


class LabelArray(BaseArray):

  def __init__(self, data):
    BaseArray.__init__(self, data)

  def processLable(self):
    problem_indexes = []
    self.labels = []
    for i, l in enumerate(self.data):
      if not isinstance(l, str):
        problem_indexes.append(i)
      try:
        temp = self.replace(l.lower())
      except:
        problem_indexes.append(i)
        self.labels.append('__EMPTY__')
        continue
      else:
        temp = temp.split(';')
        if len(temp) != 1:
          problem_indexes.append(i)
          self.labels.append('__MULTIPLE__')
        else:
          self.labels.append(temp[0])
    self.labels = np.array(self.labels)
    assert len(self.data) == len(self.labels)
    print('There are {} problematic labels'.format(len(problem_indexes)))
    return problem_indexes


class EmailArray(BaseArray):

  def __init__(self, data):
    BaseArray.__init__(self, data)

  def preprocessEmail(self):
    problem_indexes = []
    self.emails = []
    for i, e in enumerate(self.data):
      try:
        temp = re.sub('\S+@\S+', '__EMAILADDRESS__', e)
      except TypeError:
        problem_indexes.append(i)
        self.emails.append('__EMPTY__')
      else:
        self.emails.append(e)
    self.emails = np.array(self.emails)
    assert len(self.data) == len(self.emails)
    print('There are {} problematic emails'.format(len(problem_indexes)))
    return problem_indexes


def removeProblemIndexes(save_file_prefix, label_array, email_array,
                         indexes_to_remove):
  label_array = np.delete(label_array, indexes_to_remove)
  email_array = np.delete(email_array, indexes_to_remove)
  assert len(label_array) == len(email_array)
  save_file_dir = os.path.dirname(save_file_prefix)
  if not os.path.exists(save_file_dir):
    os.mkdir(save_file_dir)
  np.save(save_file_prefix + '_labels.npy', label_array)
  np.save(save_file_prefix + '_emails.npy', email_array)
  print('labels length: {}, emails length: {}'.format(len(label_array),
                                                      len(email_array)))
  print(Counter(label_array))
  return label_array, email_array


def filterByLabel(save_file_prefix, label_array, email_array, labels):
  indexes = [i for i, l in enumerate(label_array) if l in labels]
  label_array = label_array[indexes]
  email_array = email_array[indexes]
  file_name_prefix = '_'.join(labels)
  dirname = os.path.dirname(save_file_prefix)
  np.save(dirname + '/' + file_name_prefix + '_labels.npy', label_array)
  np.save(dirname + '/' + file_name_prefix + '_emails.npy', email_array)
  print('after filtering, labels length: {}, emails length: {}'.format(
      len(label_array), len(email_array)))
  print(Counter(label_array))
  return label_array, email_array


def balanceSplitDataset(split_data_path,
                        label_array,
                        email_array,
                        sampling_num=None,
                        split_ratio=0.2):
  cats, counts = np.unique(label_array, return_counts=True)
  cat_count = {cat: count for (cat, count) in zip(cats, counts)}
  print(cat_count)
  sampling_num = sampling_num if sampling_num else min(counts)
  indexes = {}
  for cat in cats:
    temp_indexes = np.array([i for i, l in enumerate(label_array) if l == cat
                            ][:sampling_num])
    indexes[cat] = temp_indexes
  labels = np.concatenate([label_array[indexes[c]] for c in cats])
  emails = np.concatenate([email_array[indexes[c]] for c in cats])

  shuffler = np.random.permutation(sampling_num * 2)
  print(shuffler)
  labels, emails = labels[shuffler], emails[shuffler]

  cutoff = int(sampling_num * 2 * split_ratio)
  train_label, valid_label = labels[cutoff:], labels[:cutoff]
  train_email, valid_email = emails[cutoff:], emails[:cutoff]

  np.save(split_data_path + 'train_label.npy', train_label)
  np.save(split_data_path + 'train_email.npy', train_email)
  np.save(split_data_path + 'valid_label.npy', valid_label)
  np.save(split_data_path + 'valid_email.npy', valid_email)

  print("Train size {}, Test size {}".format(len(train_label),
                                             len(valid_label)))
  assert len(train_email) == len(train_label)
  assert len(valid_email) == len(valid_label)
  print("Training data distribution is {}".format(Counter(train_label)))
  print("Valid data distribution is {}".format(Counter(valid_label)))
  return train_email, train_label, valid_email, valid_label


if __name__ == '__main__':

  save_file_prefix = './processing_pipeline_exprmt_20200630/json_exprmt/json_new_update'
  label_data = np.load(
      'processing_pipeline_exprmt_20200630/json_exprmt/json_all_labels.npy',
      allow_pickle=True)
  email_data = np.load(
      'processing_pipeline_exprmt_20200630/json_exprmt/json_all_bodies.npy',
      allow_pickle=True)

  la = LabelArray(label_data)
  idx_l = la.processLable()

  ea = EmailArray(email_data)
  idx_e = ea.preprocessEmail()

  indexes_to_remove = np.array(list(set(idx_l + idx_e)))

  label_array, email_array = removeProblemIndexes(save_file_prefix, la.labels,
                                                  ea.emails, indexes_to_remove)
  label_array, email_array = filterByLabel(save_file_prefix,
                                           label_array,
                                           email_array,
                                           labels=['update', 'new'])
  split_data_path = './processing_pipeline_exprmt_20200630/json_exprmt/'
  train_email, train_label, valid_email, valid_label = balanceSplitDataset(
      split_data_path, label_array, email_array)
