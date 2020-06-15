# take a look at the augmented data
# find potential bugs -> imcompleted sentences, messy meaning
# unknown1 = 426 + 1000  paraphrases
# unknown2 = 426 + 2100 paraphrases
# wider network | deeper network | wider and deeper network

import numpy as np
import pdb
# check data ----------------------------------------------
# check 600
# check 1500
# paraphrases_600 = np.load('processed_data/paraphrases.npy')
# paraphrases_1500 = np.load('processed_data/paraphrases_dbl.npy')

# for p in paraphrases_1500:
#   p = p.encode("utf-8")
#   print(p, '\n')

# append and randomize data ----------------------------------------------
# total 2100 paraphrases (600 + 1500)
# 426 + 1000 unknown

# append data
# appen labels


def composeData(org_sents, sents, org_labels, labels, *args):
  # append data and label

  new_sents = np.append(org_sents, sents)
  new_labels = np.append(org_labels, labels, axis=0)
  if args:
    if len(args) == 2:
      if isinstance(args[0][0], str):
        new_sents = np.append(new_sents, args[0])
        new_labels = np.append(new_labels, args[1])
      else:
        new_sents = np.append(new_sents, args[1])
        new_labels = np.append(new_labels, args[0])
    else:
      print('expected flexible args as a pair')
      raise AttributeError

  return new_sents, new_labels


def randomizeData(sents, labels, file_suffix=None):
  # randomize data and label
  perms = np.random.permutation(len(sents))
  new_sents = sents[perms]
  new_labels = labels[perms]
  np.save('processed_data/permutation' + file_suffix + '.npy', perms)
  np.save('processed_data/randomized_sents' + file_suffix + '.npy', new_sents)
  np.save('processed_data/randomized_labels' + file_suffix + '.npy', new_labels)


if __name__ == '__main__':
  org_data = np.load('./data/train_sents_mixed.npy', allow_pickle=True)
  org_label = np.load('./data/train_labels_onehot_mixed.npy')
  print(len(org_data))

  # 430 + 1000
  paraphrases = np.load('processed_data/paraphrases_dbl.npy')[:1000]
  paraphrases_labels = np.load(
      'processed_data/paraphrases_labels_dbl.npy')[:1000]
  new_sents, new_labels = composeData(org_data, paraphrases, org_label,
                                      paraphrases_labels)
  print(len(new_sents))
  randomizeData(new_sents, new_labels, '_1k')
  # 430 + 2100
  paraphrases = np.load('processed_data/paraphrases_dbl.npy')
  print(len(paraphrases))
  paraphrases_labels = np.load('processed_data/paraphrases_labels_dbl.npy')
  paraphrases_s = np.load('processed_data/paraphrases.npy')
  paraphrases_labels_s = np.load('processed_data/paraphrases_labels.npy')
  print(len(paraphrases_s))

  new_sents, new_labels = composeData(org_data, paraphrases, org_label,
                                      paraphrases_labels, paraphrases_s,
                                      paraphrases_labels_s)
  randomizeData(new_sents, new_labels, '_2k')
  print(len(new_sents))

# |--------------|--------------|
# |     intent   |  numbers     |
# |--------------|--------------|
# |     'UPDATE' |     4650     |
# |--------------|--------------|
# |     'NEW'    |      1495    |
# |--------------|--------------|
# |    'UNKNOWN' |      510     |
# |--------------|--------------|
