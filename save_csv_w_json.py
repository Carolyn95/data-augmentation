# csv -> remove empty and multi labels
# jsoncsv -> append file name as label

import pandas as pd
import numpy as np
import pickle as pkl
import pdb
import re

new_df = pd.read_csv('data/NEW.csv')
new_df = np.array(new_df.EmailContent)
new_df_labels = np.array(['New'] * len(new_df))
unknown_df = pd.read_csv('data/UNKNOWN.csv')
unknown_df = np.array(unknown_df.EmailContent)
unknown_df_labels = np.array(['Unknown'] * len(unknown_df))
update_df = pd.read_csv('data/UPDATE.csv')
update_df = np.array(update_df.EmailContent)
update_df_labels = np.array(['Update'] * len(update_df))

emails_dec = np.load('./data/dec_bodies.npy', allow_pickle=True)
emails_jan = np.load('./data/jan_bodies.npy', allow_pickle=True)
labels_dec = np.load('./data/dec_intentions.npy', allow_pickle=True)
labels_jan = np.load('./data/jan_intentions.npy', allow_pickle=True)

adict = {
    'unknown': 'unknown',
    'update': 'update',
    'new': 'new',
    'resolved': 'resolved'
}


def removeByLabelIndex(labels, emails):

  label_regex = re.compile("|".join(map(re.escape, adict.keys())))

  def replace(regex, text):
    return ';'.join(regex.findall(text))

  def getIndexLabel(label_array):
    labels_p = []
    index_from_label = []
    for i, la in enumerate(label_array):
      if not isinstance(la, str):
        continue

      processed_label = replace(label_regex, la.lower()).strip().split(';')

      if len(processed_label) == 1 and processed_label != ['']:
        labels_p.append(processed_label)
        index_from_label.append(i)
    return index_from_label, labels_p

  index_from_label, labels_ = getIndexLabel(labels)
  labels = np.array([label[0] for label in labels_])
  emails = emails[index_from_label]
  print(len(labels))
  assert len(labels) == len(emails)
  return labels, emails


labels_dec, emails_dec = removeByLabelIndex(labels_dec, emails_dec)
labels_jan, emails_jan = removeByLabelIndex(labels_jan, emails_jan)

all_emails = np.concatenate(
    (new_df, unknown_df, update_df, emails_dec, emails_jan))
all_labels = np.concatenate((new_df_labels, unknown_df_labels, update_df_labels,
                             labels_dec, labels_jan))

compose_df = pd.DataFrame(columns=['Emails', 'Intents'])
compose_df.Emails = all_emails
compose_df.Intents = all_labels

compose_df.to_csv('processed_data/All.csv')
