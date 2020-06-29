"""
key difference is rfind and rindex
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


class DataProcessor():

  def __init__(self,
               adict=ADICT,
               ending_pattern=ENDING_PATTERN,
               greeting_word=GREETING_WORD):

    labels_1 = np.load('./org_array_data/dec_intentions.npy', allow_pickle=True)
    labels_2 = np.load('./org_array_data/jan_intentions.npy', allow_pickle=True)
    emails_1 = np.load('./org_array_data/dec_bodies.npy', allow_pickle=True)
    emails_2 = np.load('./org_array_data/jan_bodies.npy', allow_pickle=True)
    self.labels = np.concatenate([labels_1, labels_2])
    self.emails = np.concatenate([emails_1, emails_2])
    assert len(self.labels) == len(self.emails)
    # self.data
    self.label_regex = re.compile("|".join(map(re.escape, adict.keys())))
    self.email_regex = re.compile("(?=(" +
                                  "|".join(map(re.escape, ending_pattern)) +
                                  "))")
    self.greeting_word = greeting_word
    self.sig_model = fasttext.load_model('../sig_model/model_sigclf.bin')

  def removeByLabelIndex(self):

    def replace(regex, text):
      return ';'.join(regex.findall(text))

    def getIndexLabel(label_array):
      labels_p = []
      index_from_label = []
      for i, la in enumerate(label_array):
        if not isinstance(la, str):
          continue

        processed_label = replace(self.label_regex,
                                  la.lower()).strip().split(';')

        if len(processed_label) == 1 and processed_label != ['']:
          labels_p.append(processed_label)
          index_from_label.append(i)
      return index_from_label, labels_p

    self.index_from_label, self.labels = getIndexLabel(self.labels)
    self.emails = self.emails[self.index_from_label]
    print(len(self.labels), len(self.emails))
    assert len(self.labels) == len(self.emails)

  def removeByEmailIndex(self):
    self.email_bodies = []
    self.email_body_length = []
    # train_sigclf = []
    remove_email_index = []
    # long_email_index = []
    count_long, count_empty = 0, 0

    def decomposeEmailToSentence(email_content: str) -> list:
      email_body_list = list(
          filter(None, [_.strip() for _ in email_content.split('\n')]))
      try:
        email_sents = reduce(operator.concat,
                             [re.split('[.!?]', _) for _ in email_body_list])
        # email_sents = reduce(operator.concat, [_ for _ in email_body_list])
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

    for i, email_text in enumerate(self.emails):
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
        ending_word_index = email_text.lower().rfind(ending_word_index)
      except IndexError:
        ending_word_index = len(email_text)
      email_content = email_text[:ending_word_index]
      email_sents = decomposeEmailToSentence(email_content)
      sig_clf_result = self.sig_model.predict(email_sents)[0]
      sig_clf_pred = [_[0] for _ in sig_clf_result]
      try:
        sig_start_idx = max(loc for loc, val in enumerate(sig_clf_pred)
                            if val == '__label__SIG')
      except ValueError:
        sig_start_idx = len(sig_clf_pred)
      email_sents_body = email_sents[:sig_start_idx]
      email_sents_body = removeGreeting(email_sents_body)
      length_ = len(email_sents_body)
      f = open("./data_v2/long_email_1000.txt", "a")
      if length_ > 1000:  # 32 emails are removed
        write_string = str(length_) + '\n'
        f.write(write_string)
        f.writelines(email_sents_body)
        f.write('\n')
        count_long += 1
        remove_email_index.append(i)
      elif length_ == 0:
        count_empty += 1
        remove_email_index.append(i)
      else:
        self.email_body_length.append(length_)
        self.email_bodies.append(email_sents_body)
    print(count_long, count_empty)
    f.close()

    def removeLabelByEmptyEmailIndex(labelArray, indexToRemove):
      return np.delete(labelArray, indexToRemove)

    self.labels_p = removeLabelByEmptyEmailIndex(self.labels,
                                                 remove_email_index)
    print(len(self.labels_p), len(self.email_bodies))
    assert len(self.labels_p) == len(self.email_bodies)
    self.email_bodies = [' '.join(b) for b in self.email_bodies]
    np.save('./data_v2/csv_feed_labels.npy', self.labels_p)
    np.save('./data_v2/csv_feed_emails.npy', self.email_bodies)


if __name__ == '__main__':
  dp = DataProcessor(ADICT, ENDING_PATTERN, GREETING_WORD)
  dp.removeByLabelIndex()
  dp.removeByEmailIndex()
