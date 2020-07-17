import numpy as np
import pdb
import pandas as pd
import re


## json: new and update only
new_path = "data/NEW.csv"
update_path = "data/UPDATE.csv"

new_df = pd.read_csv(new_path)
update_df = pd.read_csv(update_path)

new_sents = np.array(new_df.EmailContent)
new_sents = [re.sub('\S+@\S+', '__EMAILADDRESS__', e) for e in new_sents]

update_sents = np.array(update_df.EmailContent)
update_sents = [re.sub('\S+@\S+', '__EMAILADDRESS__', e) for e in update_sents]

new_labels = np.array([l.lower() for l in new_df.EmailIntent])
update_labels = np.array([l.lower() for l in update_df.EmailIntent])

sents = np.concatenate((new_sents, update_sents))
labels = np.concatenate((new_labels, update_labels))

np.save("processed_data/json_bodies.npy", sents)
np.save("processed_data/json_labels.npy", labels)

print('sents length is {}, labels length is {}'.format(len(sents), len(labels)))


## json: 5 splits
from EmailDataFactory import balanceSplitDataset
for i in range(5):
  split_data_path = 'processed_data/split_' + str(i) + '/'
  prefix = 'json_'
  balanceSplitDataset(split_data_path, prefix, labels, sents)
