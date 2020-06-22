# read data and save a csv for boss to review
# {0: "unknown", 1: "update", 2: "new"}
# save a train, save a valid
import numpy as np
import pandas as pd
from collections import Counter
import pdb

mappings = {0: "unknown", 1: "update", 2: "new"}
train_sents = np.load('processed_data/randomized_sents_1k.npy',
                      allow_pickle=True)
train_labels = np.load('processed_data/randomized_labels_1k.npy')
train_labels = np.argmax(train_labels, axis=1)
train_labels_name = [mappings[i] for i in train_labels]
print(Counter(train_labels_name))
df_1k = pd.DataFrame(columns=['emails', 'intentions'])
df_1k.emails = train_sents
df_1k.intentions = train_labels_name
df_1k.to_csv('processed_data/train.csv')

valid_sents = np.load('data/valid_sents_mixed.npy', allow_pickle=True)
valid_labels = np.load('data/valid_labels_onehot_mixed.npy')
valid_labels = np.argmax(valid_labels, axis=1)
valid_labels_name = [mappings[i] for i in valid_labels]
print(Counter(valid_labels_name))
df_2k = pd.DataFrame(columns=['emails', 'intentions'])
df_2k.emails = valid_sents
df_2k.intentions = valid_labels_name
df_2k.to_csv('processed_data/valid.csv')
