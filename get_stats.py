"""
{0: 'unknow', 1: 'update', 2: 'new'}
"""

import numpy as np
from collections import Counter
import pandas as pd
import pdb
from functools import reduce
import operator

total_label = np.load('./data/train_labels_onehot_s.npy', allow_pickle=True)
total_email = np.load('./data/train_sents_s.npy', allow_pickle=True)
total_label = np.argmax(total_label, axis=1)

total_label_dict = Counter(total_label)
total_indexes = []
total_values = []
for k, v in total_label_dict.items():
  total_indexes.append(k)
  total_values.append(v)

df_overview = pd.DataFrame(columns=['intentions', 'numbers'])
df_overview.intentions = total_indexes
df_overview.numbers = total_values
print(df_overview)
import pdb
pdb.set_trace()
total = []
for e, l in zip(total_email, total_label):
  temp = {}
  temp['content'] = e
  temp['label'] = l
  total.append(temp)

update_ = [e['content'] for e in total if e['label'] == 'update']
new_ = [e['content'] for e in total if e['label'] == 'new']
unknow_ = [e['content'] for e in total if e['label'] == 'unknown']


def getStats(anarray, intention):

  sent_num = [len(_) for _ in anarray]
  word_length = []
  for a in anarray:
    temp_word_length = [len(_.split()) for _ in a]
    word_length.append(temp_word_length)
  word_count_per_sent = reduce(operator.concat, word_length)

  result = {}
  result['intentions'] = intention
  result['max_sent_num'] = max(sent_num)
  result['min_sent_num'] = min(sent_num)
  result['avg_sent_num'] = np.mean(sent_num)
  result['max_word_count'] = max(word_count_per_sent)
  result['min_word_count'] = min(word_count_per_sent)
  result['avg_word_count'] = np.mean(word_count_per_sent)

  return result


update_result = getStats(update_, "update")
new_result = getStats(new_, "new")
unknow_result = getStats(unknow_, "unknow")
d = []
d.append(update_result)
d.append(new_result)
d.append(unknow_result)
df_sent_stats = pd.DataFrame(d)
print(df_sent_stats)