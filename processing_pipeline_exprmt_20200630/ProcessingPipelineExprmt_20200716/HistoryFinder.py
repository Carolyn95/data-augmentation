import numpy as np
import pdb
import pickle as pkl
import re
import os
import operator
from pathlib import Path

TEST = 'TEST'


class HistoryFinder():

  def __init__(self, data_dir, sents, labels, prefix):
    self.emails = sents
    self.labels = labels
    self.file_prefix = data_dir + prefix

  def parseRawMessage(self):

    def decomposeEmailToSentence(email_content: str) -> list:
      email_sents = list(
          filter(None, [_.strip() for _ in email_content.split('\n')]))
      return email_sents

    self.emails_p = []
    for i, e in enumerate(self.emails):
      email_text = e.replace(u'\xa0', u' ').replace('&#8217;', '\'').replace(
          '\\r\\n', '\n').replace('\u3000', '')
      start_of_conversation = email_text.find('\nFrom:')
      first_thread_email = decomposeEmailToSentence(
          email_text[:start_of_conversation])

      self.emails_p.append(' '.join(first_thread_email))

    self.emails_p = np.array(self.emails_p)
    np.save(self.file_prefix + '_email.npy', self.emails_p)
    np.save(self.file_prefix + '_label.npy', self.labels)
    print('# of emails after processing: {} '.format(len(self.emails_p)))


if __name__ == '__main__':

  parent_dir = './processed_data/'
  serial_no = 'split_0/'
  save_dir = parent_dir + serial_no

  train_email = np.load(save_dir + 'train_emails.npy', allow_pickle=True)
  train_label = np.load(save_dir + 'train_labels.npy', allow_pickle=True)
  valid_email = np.load(save_dir + 'valid_emails.npy', allow_pickle=True)
  valid_label = np.load(save_dir + 'valid_labels.npy', allow_pickle=True)

  ea_train = HistoryFinder(save_dir, train_email, train_label, 'KW_train')
  ea_train.parseRawMessage()
  ea_valid = HistoryFinder(save_dir, valid_email, valid_label, 'KW_valid')
  ea_valid.parseRawMessage()
