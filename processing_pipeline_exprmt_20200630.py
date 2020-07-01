"""
# 3 entries
# data reader
# data processor
# classifier

# ensure same split in different processing pipeline are using the same set of data
# the only difference among group 13 | 14 | 15 is processing pipeline
# the difference among different splits in one group is the split
"""

from pathlib import Path
import os
import numpy as np
import pandas as pd
from collections import Counter
import pdb
import time
from EmailDataFactory import filterByLabel, balanceSplitDataset
from ScoopDataProcessor import ScoopDataProcessor
from HistoryFinder import HistoryFinder
from RevScoopDataProcessor import RevScoopDataProcessor
from IntentClassifier import ModelDataReader, VanillaUSE


class DataReader():
  # create split here
  # save into different folders
  def __init__(self, save_path, **kwargs):
    self.save_path = save_path
    if not os.path.exists(save_path):
      os.mkdir(save_path)

  def preprocess(self, labels, emails, filtered_labels):
    label_array, email_array = filterByLabel(self.save_path,
                                             labels,
                                             emails,
                                             labels=filtered_labels)
    train_email, train_label, valid_email, valid_label = balanceSplitDataset(
        self.save_path, label_array, email_array)

    return train_email, train_label, valid_email, valid_label


class DataProcessor():
  # different processing pipeline here
  def __init__(self, save_dir):
    self.save_dir = save_dir

  def cleanBySigClf(self, train_email, valid_email, train_label, valid_label):
    train_scp = ScoopDataProcessor(self.save_dir)
    train_scp.removeByEmailIndex(train_email, train_label, 'train')
    valid_scp = ScoopDataProcessor(self.save_dir)
    valid_scp.removeByEmailIndex(valid_email, valid_label, 'valid')

  def cleanByKW(self, train_email, valid_email, train_label, valid_label):
    train_hf = HistoryFinder(self.save_dir, train_email, train_label,
                             'KW_train')
    train_hf.parseRawMessage()
    valid_hf = HistoryFinder(self.save_dir, valid_email, valid_label,
                             'KW_valid')
    valid_hf.parseRawMessage()

  def cleanBySigClfRev(self, train_email, valid_email, train_label,
                       valid_label):
    train_rscp = RevScoopDataProcessor(self.save_dir)
    train_rscp.removeByEmailIndex(train_email, train_label, 'train')
    valid_rscp = RevScoopDataProcessor(self.save_dir)
    valid_rscp.removeByEmailIndex(valid_email, valid_label, 'valid')


class Classifier():
  # moddeling
  def __init__(self):
    print()


if __name__ == '__main__':
  parent_dir = 'processing_pipeline_exprmt_20200630/'
  serial_no = 'split_4/'
  data_dir = parent_dir + 'trytry/'
  save_dir = parent_dir + serial_no

  # Step 1
  # all_labels = np.load(data_dir + 'all__labels.npy', allow_pickle=True)
  # all_emails = np.load(data_dir + 'all__emails.npy', allow_pickle=True)
  # dr = DataReader(save_dir)
  # dr.preprocess(all_labels, all_emails, ['update', 'new'])

  # Step 2
  # train_label = np.load(save_dir + 'train_label.npy', allow_pickle=True)
  # valid_label = np.load(save_dir + 'valid_label.npy', allow_pickle=True)
  # train_sent = np.load(save_dir + 'train_email.npy', allow_pickle=True)
  # valid_sent = np.load(save_dir + 'valid_email.npy', allow_pickle=True)

  # dp = DataProcessor(save_dir)
  # dp.cleanBySigClf(train_sent, valid_sent, train_label, valid_label)
  # dp.cleanByKW(train_sent, valid_sent, train_label, valid_label)
  # dp.cleanBySigClfRev(train_sent, valid_sent, train_label, valid_label)

  # Step 3
  start_time = time.time()
  prefix = 'scp_'  # 'KW_' | 'rev_scp_' | 'scp_'
  input_path = Path(save_dir)
  train_sents_path = input_path / (prefix + 'train_email.npy')
  train_labels_path = input_path / (prefix + 'train_label.npy')
  valid_sents_path = input_path / (prefix + 'valid_email.npy')
  valid_labels_path = input_path / (prefix + 'valid_label.npy')
  mdr = ModelDataReader(prefix, train_sents_path, train_labels_path,
                        valid_sents_path, valid_labels_path)
  mdr.getStats()
  mdr.onehotEncodingLabel(input_path)
  mdr.listToStr()
  mdr.randomizeData(input_path)

  vu = VanillaUSE(mdr.train_sents, mdr.onehot_train_labels, mdr.valid_sents,
                  mdr.onehot_valid_labels)
  vu.createModel()
  vu.train(filepath=str(input_path) + '/' + prefix + 'model')
  vu.consolidateResult(str(input_path) + '/' + prefix + 'model')
  print('Overall Time: ', str(time.time() - start_time), 's')
