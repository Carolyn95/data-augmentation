# version 0: my preprocessing pipeline
# version 0 - improvement: word distribution to cut
# version 1: use original email, ?tokenization? but retain lastest email by FROM: Sent: To: only
# version 2: reverse signature finding order

# csv + json

# obvious problem in my pipeline:
# 1, only "hi," "hi, name", "name" left
# 2, list of characters left, "M e e t i n g"
# 3, sentences got chopped off

# save train.csv & test.csv _v1 to compare


"""
## process labels:
# - remove naturally empty 'nan' (means, these records are empty in original data)
# - save into 'empty_labels.pkl'
# - process label by RegEx
# - remove multi-labels from processed labels
# - save multi-labels indexes in 'mul_labels.pkl'
# - remove empty labels from processed and remove multi-labels array
# - save processed empty in 'processed_empty.pkl'
# - final output saves in 'intentions_singular.npy'

## process email_body: 
# - remove nan (originally empty records)
# - remove multi-labels records
# - remove processed empty records
### dec has empty emails: index in org_data: [647, 648, 649, 650, 651, 1378]
### CHANGES: compare before, separate signature by rules this time, take [-1] instead of [0]
### CHANGES: try catch errors, coz email could be empty

# ~650 empty
"""

import numpy as np
import pdb
import pickle as pkl
import re
import os
from functools import reduce
import operator
import fasttext


class BaseArray():

  def __init__(self, data_dir, file_name):
    self.data = np.load(data_dir + '/' + file_name, allow_pickle=True)
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

  def removeEmpty(self):
    empty_labels = [
        i for i, l in enumerate(self.data) if not isinstance(l, str)
    ]
    with open(data_dir + '/' + prefix + 'empty_labels.pkl', 'wb') as f:
      pkl.dump(empty_labels, f)
    self.labels = np.delete(self.data, empty_labels)

  def processLabel(self):
    self.labels_p = [self.replace(t.lower()) for t in self.labels]
    assert len(self.labels_p) == len(self.labels)

    new_labels = [l.split(';') for l in self.labels_p]
    new_labels_length = [len(l) for l in new_labels]
    mul_labels = [i for i, nl in enumerate(new_labels_length) if nl > 1]
    with open(data_dir + '/' + self.prefix + 'mul_labels.pkl', 'wb') as f:
      pkl.dump(mul_labels, f)
    self.labels_p = np.delete(self.labels_p, mul_labels)

    processed_empty = [i for i, l in enumerate(self.labels_p) if not l]

    with open(data_dir + '/' + self.prefix + 'processed_empty.pkl', 'wb') as f:
      pkl.dump(processed_empty, f)

    self.labels_p = np.delete(self.labels_p, processed_empty)
    np.save(data_dir + '/' + self.prefix + 'intentions_singular.npy',
            self.labels_p)
    print(len(self.labels_p))


class EmailArray(BaseArray):

  def __init__(self, data_dir, file_name, prefix):
    BaseArray.__init__(self, data_dir, file_name)
    # self.prefix = prefix  # dec_
    self.file_prefix = data_dir + '/' + prefix
    self.sig_model = fasttext.load_model('../sig_model/model_sigclf.bin')
    self.ending_patterns = [
        '\nregards', '\nbest', '\nthank', '\nthanks', '\nrgds', '\ntks',
        '\nbrgds', '\ncheers', '\nyours', '\nsincerely', '\nthks',
        '\nwarm regards', '\nthank you', '\nbest regards', '\nyours sincerely'
    ]
    self.greeting_words = ["hi", "hello", "dear", "good"]
    self.regex = re.compile("(?=(" +
                            "|".join(map(re.escape, self.ending_patterns)) +
                            "))")

  def matchEmails(self):

    with open(self.file_prefix + 'empty_labels.pkl', 'rb') as f:
      nan_idx = pkl.load(f)
    with open(self.file_prefix + 'mul_labels.pkl', 'rb') as f:
      mul_idx = pkl.load(f)
    with open(self.file_prefix + 'processed_empty.pkl', 'rb') as f:
      emp_idx = pkl.load(f)

    # - remove nan (originally empty records)
    self.emails = np.delete(self.data, nan_idx)
    # - remove multi-labels records
    self.emails = np.delete(self.emails, mul_idx)
    # - remove processed empty records
    self.emails = np.delete(self.emails, emp_idx)
    print(len(self.emails))

  def separateSignature(self):
    self.email_bodies = []
    train_sigclf = []

    for i, email_text in enumerate(self.emails):
      # split by rules
      self.temp_ = email_text
      try:
        email_text = re.sub('\S+@\S+', '__EMAILADDRESS__', email_text)
      except TypeError:
        email_text = '__EMPTYEMAIL__'
      email_text = email_text.replace(u'\xa0',
                                      u' ').replace('&#8217;', '\'').replace(
                                          '\\r\\n', '\n').replace('\u3000', '')

      try:
        ending_word_index = re.findall(self.regex, email_text.lower())[-1]
        ending_word_index = email_text.lower().find(ending_word_index)
      except IndexError:
        ending_word_index = len(email_text)

      email_content = email_text[:ending_word_index]
      # save for further training signature classifier usage
      temp = {}
      temp['content'] = email_content
      temp['signature_block'] = email_text[ending_word_index:]
      train_sigclf.append(temp)

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
            if t.lower() in self.greeting_words:
              greeting.append(line)
        if len(lines) > 1:
          lines = [l for l in lines if l not in greeting]
        return lines

      email_sents = decomposeEmailToSentence(email_content)
      email_sents = removeGreeting(email_sents)
      # split by signature classifier
      sig_clf_result = self.sig_model.predict(email_sents)[0]
      sig_clf_pred = [_[0] for _ in sig_clf_result]
      try:
        sig_start_idx = sig_clf_pred.index('__label__SIG')
      except ValueError:
        sig_start_idx = len(sig_clf_pred)
      email_sents = email_sents[:sig_start_idx]
      self.email_bodies.append(email_sents)
    self.email_bodies = np.array(self.email_bodies)
    np.save(self.file_prefix + 'email_bodies.npy', self.email_bodies)




if __name__ == '__main__':
  # 2805 => 2790 => 2770
  # data_dir = 'processed_data'
  # file_name = 'dec_intentions.npy'
  # prefix = 'dec_'

  # 4037 => 4025 => 3987
  # data_dir = 'processed_data'
  # file_name = 'jan_intentions.npy'
  # prefix = 'jan_'

  # la = LabelArray(data_dir, file_name, prefix)
  # la.removeEmpty()
  # la.processLabel()

  # data_dir = 'processed_data'
  # file_name = 'dec_bodies.npy'
  # prefix = 'dec_'

  data_dir = 'processed_data'
  file_name = 'jan_bodies.npy'
  prefix = 'jan_'

  ea = EmailArray(data_dir, file_name, prefix)
  ea.matchEmails()
  ea.separateSignature()
