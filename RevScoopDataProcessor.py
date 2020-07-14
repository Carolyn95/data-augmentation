"""
key difference is rfind and max(loc for loc, val in enumerate(sig_clf_pred)
                            if val == '__label__SIG')
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


class RevScoopDataProcessor():

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
    self.sig_model = fasttext.load_model('../sig_model/model_sigclf.bin')

  def removeByEmailIndex(self, emails, labels, prefix):
    print('-------rscp-------', prefix)
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

    for i, email_text in enumerate(emails):
      try:
        email_text = re.sub('\S+@\S+', '__EMAILADDRESS__', email_text)
      except TypeError:
        remove_email_index.append(i)
        continue
      email_text = email_text.replace(u'\xa0', u' ').replace(
          '&#8217;', '\'').replace('\\r\\n',
                                   '\n').replace('\u3000',
                                                 '').replace('\r\n', '\n')
      start_of_conversation = email_text.find('\nFrom:')
      email_text = email_text[:start_of_conversation]
      try:
        ending_word_index = re.findall(self.email_regex, email_text.lower())[-1]
        ending_word_index = email_text.lower().rfind(ending_word_index)
      except IndexError:
        ending_word_index = len(email_text)
      email_content = email_text[:ending_word_index]
      email_sents = decomposeEmailToSentence(email_content)
      sig_clf_result = self.sig_model.predict(email_sents)[0]
      sig_clf_pred = [_[0] for _ in sig_clf_result]
      sig_index = [
          loc for loc, val in enumerate(sig_clf_pred) if val == '__label__SIG'
      ]
      sig_start_idx = max(sig_index) if len(sig_index) > 0 and max(
          sig_index) != 0 else len(sig_clf_pred)
      email_sents_body = email_sents[:sig_start_idx]
      length_ = len(email_sents_body)

      if length_ > 1000:
        count_long += 1
        remove_email_index.append(i)
      elif length_ == 0:
        count_empty += 1
        remove_email_index.append(i)
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
    np.save(self.data_dir + 'rev_scp_' + prefix + '_label.npy', self.labels_p)
    np.save(self.data_dir + 'rev_scp_' + prefix + '_email.npy',
            self.email_bodies)


if __name__ == '__main__':
  parent_dir = 'processing_pipeline_exprmt_20200630/'
  serial_no = 'json_exprmt/'
  data_dir = parent_dir + serial_no
  train_label = np.load(data_dir + 'train_label.npy', allow_pickle=True)
  valid_label = np.load(data_dir + 'valid_label.npy', allow_pickle=True)
  train_email = np.load(data_dir + 'train_email.npy', allow_pickle=True)
  valid_email = np.load(data_dir + 'valid_email.npy', allow_pickle=True)

  train_rscp = RevScoopDataProcessor(data_dir)
  train_rscp.removeByEmailIndex(train_email, train_label, 'train')
  valid_rscp = RevScoopDataProcessor(data_dir)
  valid_rscp.removeByEmailIndex(valid_email, valid_label, 'valid')
