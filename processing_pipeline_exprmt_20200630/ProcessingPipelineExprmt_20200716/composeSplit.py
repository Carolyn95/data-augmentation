# compose json & dec & jan, train into one
# compose json & dec & jan, valid into one
import numpy as np 
import pdb


def composeSplit(folder, train_or_valid, prefix):
  
  for i, p in enumerate(prefix):
    if i == 0:
      emails = np.load(folder+p+train_or_valid+'email.npy', allow_pickle=True)
      labels = np.load(folder+p+train_or_valid+'label.npy', allow_pickle=True)
    else:
      temp_email = np.load(folder+p+train_or_valid+'email.npy', allow_pickle=True)
      temp_label = np.load(folder+p+train_or_valid+'label.npy', allow_pickle=True)
      emails = np.concatenate((emails, temp_email))
      labels = np.concatenate((labels, temp_label))

  np.save(folder+train_or_valid+'emails.npy', emails)
  np.save(folder+train_or_valid+'labels.npy', labels)

folder_ = 'processed_data/split_'
flags = ['train_', 'valid_']
prefix = ['dec_', 'jan_', 'json_']

for i in range(5):
  folder = folder_ + str(i) + '/'
  for train_or_valid in flags:
    composeSplit(folder, train_or_valid, prefix)
