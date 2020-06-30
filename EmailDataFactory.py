import numpy as np
import pdb
import pickle as pkl
import re
import os
import operator
from pathlib import Path


class BaseArray():

  def __init__(self, data_dir, file_name):
    self.data = np.load(data_dir + '/' + file_name, allow_pickle=True)
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

  def __init__(self, data_dir, file_name, prefix):
    BaseArray.__init__(self, data_dir, file_name)
    self.prefix = prefix  # dec_

  def removeEmpty(self, save_dir):
    empty_labels = [
        i for i, l in enumerate(self.data) if not isinstance(l, str)
    ]
    print('# of empty labels: {}'.format(len(empty_labels)))
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(save_dir + '/' + prefix + 'empty_labels_indexes.pkl', 'wb') as f:
      pkl.dump(empty_labels, f)
    self.labels = np.delete(self.data, empty_labels)

  def processLabel(self, save_dir):
    self.labels_p = [self.replace(t.lower()) for t in self.labels]
    assert len(self.labels_p) == len(self.labels)

    new_labels = [l.split(';') for l in self.labels_p]
    new_labels_length = [len(l) for l in new_labels]
    mul_labels = [i for i, nl in enumerate(new_labels_length) if nl > 1]
    print('# of multiple labels: {}'.format(len(mul_labels)))
    with open(save_dir + '/' + self.prefix + 'mul_labels_indexes.pkl',
              'wb') as f:
      pkl.dump(mul_labels, f)
    self.labels_p = np.delete(self.labels_p, mul_labels)

    processed_empty = [i for i, l in enumerate(self.labels_p) if not l]
    print('# of processed empty labels: {}'.format(len(processed_empty)))
    # processed_empty means original labels don't contain target labels
    with open(save_dir + '/' + self.prefix + 'processed_empty_indexes.pkl',
              'wb') as f:
      pkl.dump(processed_empty, f)

    self.labels_p = np.delete(self.labels_p, processed_empty)
    np.save(save_dir + '/' + self.prefix + 'intentions_singular_indexes.npy',
            self.labels_p)
    print('# of singular labels: {}'.format(len(self.labels_p)))


class EmailArray(BaseArray):

  def __init__(self, data_dir, save_dir, file_name, prefix):
    BaseArray.__init__(self, data_dir, file_name)
    self.file_prefix = save_dir + '/' + prefix  # dec_

  def cleanByLabelIndex(self):

    with open(self.file_prefix + 'empty_labels_indexes.pkl', 'rb') as f:
      nan_idx = pkl.load(f)
    with open(self.file_prefix + 'mul_labels_indexes.pkl', 'rb') as f:
      mul_idx = pkl.load(f)
    with open(self.file_prefix + 'processed_empty_indexes.pkl', 'rb') as f:
      emp_idx = pkl.load(f)

    # - remove those without labels
    self.emails = np.delete(self.data, nan_idx)
    # - remove multi-labels
    self.emails = np.delete(self.emails, mul_idx)
    # - remove processed empty labels
    self.emails = np.delete(self.emails, emp_idx)
    print('# of emails(delete problematic labels indexes): {}'.format(
        len(self.emails)))

  def parseRawMessage(self):

    def decomposeEmailToSentence(email_content: str) -> list:
      email_sents = list(
          filter(None, [_.strip() for _ in email_content.split('\n')]))
      return email_sents

    self.emails_p = []
    self.empty_email_indexes = []
    for i, e in enumerate(self.emails):
      try:
        email_text = re.sub('\S+@\S+', '__EMAILADDRESS__', e)
      except TypeError:
        self.empty_email_indexes.append(i)
        continue
      email_text = email_text.replace(u'\xa0',
                                      u' ').replace('&#8217;', '\'').replace(
                                          '\\r\\n', '\n').replace('\u3000', '')
      start_of_conversation = email_text.find('\nFrom:')
      first_thread_email = decomposeEmailToSentence(
          email_text[:start_of_conversation])

      self.emails_p.append(' '.join(first_thread_email))

    if self.empty_email_indexes:
      self.empty_email_indexes = np.array(self.empty_email_indexes)
      np.save(self.file_prefix + 'empty_emails.npy', self.empty_email_indexes)

    self.emails_p = np.array(self.emails_p)
    np.save(self.file_prefix + 'feed_emails.npy', self.emails_p)
    print('# of emails after processing: {} '.format(len(self.emails_p)))


def cleanByEmailIndex(save_dir, prefix, label_array, empty_email_indexes,
                      email_array):
  label_array = np.delete(label_array, empty_email_indexes)
  np.save(save_dir + '/' + prefix + 'feed_labels.npy', label_array)
  print('labels length: {}, emails length: {}'.format(len(label_array),
                                                      len(email_array)))
  assert len(label_array) == len(email_array)


if __name__ == '__main__':

  # 'jan_' | 'dec_' | 'json_train_' | 'json_valid_'
  prefix = 'json_valid_'
  data_dir = 'org_array_data'
  save_dir = 'data_v1'

  label_file_name = prefix + 'intentions.npy'
  la = LabelArray(data_dir, label_file_name, prefix)
  la.removeEmpty(save_dir)
  la.processLabel(save_dir)

  email_file_name = prefix + 'bodies.npy'
  ea = EmailArray(data_dir, save_dir, email_file_name, prefix)
  ea.cleanByLabelIndex()
  ea.parseRawMessage()

  cleanByEmailIndex(save_dir, prefix, la.labels_p, ea.empty_email_indexes,
                    ea.emails_p)
