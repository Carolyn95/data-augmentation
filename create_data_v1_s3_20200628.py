import numpy as np
import pandas as pd
from collections import Counter

if __name__ == '__main__':
  prefixes = ['jan_', 'dec_', 'json_train_', 'json_valid_']

  data_dir = 'data_v1/'
  labels_suffix = ''
  emails_suffix = 'emails.npy'
  label_files = [data_dir + p + 'feed_labels.npy' for p in prefixes]
  email_files = [data_dir + p + 'feed_emails.npy' for p in prefixes]

  for i, f in enumerate(label_files):
    if i == 0:
      all_labels = np.load(f)
    else:
      all_labels = np.concatenate((all_labels, np.load(f)))

  for i, f in enumerate(email_files):
    if i == 0:
      all_emails = np.load(f, allow_pickle=True)
    else:
      all_emails = np.concatenate((all_emails, np.load(f, allow_pickle=True)))

  resolve_indexes = [i for i, l in enumerate(all_labels) if l == 'resolved']
  all_labels = np.delete(all_labels, resolve_indexes)
  all_emails = np.delete(all_emails, resolve_indexes)
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
  df.to_csv(data_dir + 'all_v1.csv')