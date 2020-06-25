#
import numpy as np
from collections import Counter
import pdb
from rich import print

train_sents = np.load('processed_data/all_sents_balanced.npy',
                      allow_pickle=True)
train_labels = np.load('processed_data/all_onthot_labels_balanced.npy')
int_labels_t = np.argmax(train_labels, axis=1)

valid_sents = np.load('data/valid_sents_mixed.npy', allow_pickle=True)
valid_labels = np.load('data/valid_labels_onehot_mixed.npy')
int_labels_v = np.argmax(valid_labels, axis=1)
print(Counter(int_labels_v))
# new and update only
# {1: 'update', 2: 'new'}

int_indexes_t = [i for i, v in enumerate(int_labels_t) if v in [1, 2]]
print(len(int_indexes_t))
train_sents = train_sents[int_indexes_t]
train_labels = int_labels_t[int_indexes_t]
train_labels_onehot = []
for i in train_labels:
  temp = np.zeros(2)
  temp[i - 1] = 1
  train_labels_onehot.append(temp)
train_labels_onehot = np.array(train_labels_onehot)
print(train_sents[:10])
print(train_labels_onehot[:10])
np.save('processed_data/train_sents_pairwise_update_new.npy', train_sents)
np.save('processed_data/train_labels_pairwise_update_new.npy',
        train_labels_onehot)

# {0: 109, 1: 980, 2: 327}
int_indexes_v_update = [i for i, v in enumerate(int_labels_v) if v == 1]
int_indexes_v_new = [i for i, v in enumerate(int_labels_v) if v == 2]
valid_sents_new = valid_sents[int_indexes_v_new]
valid_sents_update = valid_sents[int_indexes_v_update][:300]
valid_sents_ = np.concatenate((valid_sents_new, valid_sents_update))

valid_labels_new = int_labels_v[int_indexes_v_new]
valid_labels_update = int_labels_v[int_indexes_v_update][:300]
valid_labels_ = np.concatenate((valid_labels_new, valid_labels_update))
valid_labels_onehot = []
for i in valid_labels_:
  temp = np.zeros(2)
  temp[i - 1] = 1
  valid_labels_onehot.append(temp)
valid_labels_onehot = np.array(valid_labels_onehot)
print(valid_sents_[:10])
print(valid_labels_onehot[:10])
np.save('processed_data/valid_sents_pairwise_update_new.npy', valid_sents_)
np.save('processed_data/valid_labels_pairwise_update_new.npy',
        valid_labels_onehot)

""" RECORD
0.9553429027113237
599
(
    array([0.93589744, 0.97460317]),
    array([0.97333333, 0.93883792]),
    array([0.95424837, 0.95638629]),
    array([300, 327]),
)
              precision    recall  f1-score   support

      update       0.94      0.97      0.95       300
         new       0.97      0.94      0.96       327

    accuracy                           0.96       627
   macro avg       0.96      0.96      0.96       627
weighted avg       0.96      0.96      0.96       627
[[292   8]
 [ 20 307]]

 448.6 MiB~ 2655.3 MiB
Overall Time:  91.9804093837738 s

Epoch 20/20
2432/2497 [============================>.] - ETA: 0s - loss: 0.1576 - acc: 0.9367
Epoch 00020: saving model to serial-no/13/20.hdf5
2497/2497 [==============================] - 4s 2ms/sample - loss: 0.1580 - acc: 0.9363 - val_loss: 0.4331 - val_acc: 0.8176
"""
