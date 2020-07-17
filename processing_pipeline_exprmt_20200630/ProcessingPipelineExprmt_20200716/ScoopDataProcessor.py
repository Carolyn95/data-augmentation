"""
after filtering out emails with sentence number > 30
xxx data change from 6757 -> 6715 
data change from 6757 -> 6655 (after processing, some emails become empty)
"""
import numpy as np
import pdb
import pickle as pkl
import re
import os
from functools import reduce
import operator
import fasttext

ADICT = {
    'unknown': 'unknown',
    'update': 'update',
    'new': 'new',
    'resolved': 'resolved'
}
ENDING_PATTERN = [
    '\nregards', '\nbest', '\nthank', '\nthanks', '\nrgds', '\ntks', '\nbrgds',
    '\ncheers', '\nyours', '\nsincerely', '\nthks', '\nwarm regards',
    '\nthank you', '\nbest regards', '\nyours sincerely'
]
GREETING_WORD = ["hi", "hello", "dear", "good"]


def removeByIndex(anArray, index):
  newArray = np.delete(anArray, index)
  return newArray


class ScoopDataProcessor():

  def __init__(self,
               data_dir,
               adict=ADICT,
               ending_pattern=ENDING_PATTERN,
               greeting_word=GREETING_WORD):

    self.data_dir = data_dir
    self.label_regex = re.compile("|".join(map(re.escape, adict.keys())))
    self.email_regex = re.compile("(?=(" +
                                  "|".join(map(re.escape, ending_pattern)) +
                                  "))")
    self.greeting_word = greeting_word
    self.sig_model = fasttext.load_model('../../../sig_model/model_sigclf.bin')

  def removeByEmailIndex(self, emails, labels, prefix):
    print('-------scp-------', prefix)
    self.email_bodies = []
    self.email_body_length = []

    remove_email_index = []
    count_long, count_empty = 0, 0

    def decomposeEmailToSentence(email_content: str) -> list:
      email_body_list = list(
          filter(None, [_.strip() for _ in email_content.split('\n')]))
      try:
        email_sents = reduce(operator.concat,
                             [re.split('[.!?]', _) for _ in email_body_list])
      except TypeError as err:
        email_sents = ['__THANKYOUEMAIL__']

      email_sents = list(filter(None, [_.strip() for _ in email_sents]))
      return email_sents

    def removeGreeting(lines):
      greeting = []
      for line in lines:
        tokens = line.split()
        t_list = [re.sub(r'[^\w\s]', ' ', t).strip() for t in tokens]
        for t in t_list:
          if t.lower() in self.greeting_word:
            greeting.append(line)
      if len(lines) > 1:
        lines = [l for l in lines if l not in greeting]
      return lines

    for i, email_text in enumerate(emails):
      try:
        email_text = re.sub('\S+@\S+', '__EMAILADDRESS__', email_text)
      except TypeError:
        remove_email_index.append(i)
        continue
      email_text = email_text.replace(u'\xa0',
                                      u' ').replace('&#8217;', '\'').replace(
                                          '\\r\\n', '\n').replace('\u3000', '')
      try:
        ending_word_index = re.findall(self.email_regex, email_text.lower())[-1]
        ending_word_index = email_text.lower().find(ending_word_index)
      except IndexError:
        ending_word_index = len(email_text)
      email_content = email_text[:ending_word_index]
      email_sents = decomposeEmailToSentence(email_content)
      sig_clf_result = self.sig_model.predict(email_sents)[0]
      sig_clf_pred = [_[0] for _ in sig_clf_result]
      try:
        sig_start_idx = sig_clf_pred.index('__label__SIG')
      except ValueError:
        sig_start_idx = len(sig_clf_pred)
      email_sents_body = email_sents[:sig_start_idx]
      email_sents_body = removeGreeting(email_sents_body)
      length_ = len(email_sents_body)

      if length_ > 1000:  # 32 emails are removed
        count_long += 1
        remove_email_index.append(i)
      elif length_ == 0:
        email_sents_body = ['__PROCESSEDEMPTY__']
        print(i)
        count_empty += 1
        # remove_email_index.append(i)
        self.email_body_length.append(length_)
        self.email_bodies.append(email_sents_body)
      else:
        self.email_body_length.append(length_)
        self.email_bodies.append(email_sents_body)
    print('Long and empty emails counts are {}, {}'.format(
        count_long, count_empty))

    def removeLabelByEmptyEmailIndex(labelArray, indexToRemove):
      return np.delete(labelArray, indexToRemove)

    self.labels_p = removeLabelByEmptyEmailIndex(labels, remove_email_index)

    print(len(self.labels_p), len(self.email_bodies))
    assert len(self.labels_p) == len(self.email_bodies)
    np.save(self.data_dir + 'scp_' + prefix + '_label.npy',
            self.labels_p)
    np.save(self.data_dir + 'scp_' + prefix + '_email.npy',
            self.email_bodies)


if __name__ == '__main__':
  parent_dir = './processed_data/'
  serial_no = 'split_0/'
  data_dir = parent_dir + serial_no

  train_label = np.load(data_dir + 'train_labels.npy', allow_pickle=True)
  valid_label = np.load(data_dir + 'valid_labels.npy', allow_pickle=True)
  train_email = np.load(data_dir + 'train_emails.npy', allow_pickle=True)
  valid_email = np.load(data_dir + 'valid_emails.npy', allow_pickle=True)

  train_dp = ScoopDataProcessor(data_dir)
  train_dp.removeByEmailIndex(train_email, train_label, 'train')
  valid_dp = ScoopDataProcessor(data_dir)
  valid_dp.removeByEmailIndex(valid_email, valid_label, 'valid')
