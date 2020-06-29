import numpy as np
import pandas as pd
from collections import Counter
import pdb
"""
labels length is 7127, emails length is 7127
Counter({'update': 4939, 'new': 1646, 'unknown': 542})
"""

if __name__ == '__main__':
  prefixes = ['train_', 'valid_']

  data_dir = 'data_v2/'

  label_files = [
      'org_array_data/json_' + p + 'intentions.npy' for p in prefixes
  ]
  email_files = ['org_array_data/json_' + p + 'bodies.npy' for p in prefixes]

  for i, f in enumerate(label_files):
    if i == 0:
      all_labels = np.load(f)
    else:
      all_labels = np.concatenate((all_labels, np.load(f)))
      all_labels = np.array([l.lower() for l in all_labels])

  for i, f in enumerate(email_files):
    if i == 0:
      all_emails = np.load(f, allow_pickle=True)
    else:
      all_emails = np.concatenate((all_emails, np.load(f, allow_pickle=True)))

  resolve_indexes = [i for i, l in enumerate(all_labels) if l == 'resolved']
  all_labels = np.delete(all_labels, resolve_indexes)
  all_emails = np.delete(all_emails, resolve_indexes)
  csv_emails = np.load('data_v2/csv_feed_emails.npy', allow_pickle=True)
  csv_labels = np.load('data_v2/csv_feed_labels.npy', allow_pickle=True)
  all_labels = np.concatenate((all_labels, csv_labels))
  all_emails = np.concatenate((all_emails, csv_emails))

  print('labels length is {}, emails length is {}'.format(
      len(all_labels), len(all_emails)))

  print(Counter(all_labels))
  # labels length is 7134, emails length is 7134
  # Counter({'update': 4939, 'new': 1652, 'unknown': 543})
  np.save(data_dir + 'all_feed_labels.npy', all_labels)
  np.save(data_dir + 'all_feed_emails.npy', all_emails)

  df = pd.DataFrame(columns=['Emails', 'Intentions'])
  df.Emails = all_emails
  df.Intentions = all_labels
  df.to_csv(data_dir + 'all_v2.csv')