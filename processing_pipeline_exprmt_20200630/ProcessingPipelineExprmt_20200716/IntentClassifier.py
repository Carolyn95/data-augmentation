from collections import Counter
import re
import os
import pdb
import numpy as np
import pickle as pkl
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_hub as hub
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Input, Lambda, Dense, Dropout, Reshape, BatchNormalization, ReLU, LayerNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.optimizers.schedules import LearningRateSchedule
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import random
import time
from memory_profiler import profile
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from rich import print
import pdb
from matplotlib import pyplot as plt
import numpy as np
# from clr_callback import CyclicLR
import clr
# import tensorflow_addons as tfa
# from tensorflow_addons.optimizers import CyclicalLearningRate

import matplotlib.pyplot as plt
import numpy as np
from spacy.lang.en import English
from pathlib import Path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove logging info
import warnings
warnings.filterwarnings('ignore')  # filter out warnings
tf.random.set_random_seed(2020)


class ModelDataReader():

  def __init__(self, prefix, train_sents, train_labels, valid_sents,
               valid_labels):
    print('----- Modelling', prefix, '-----')
    self.prefix = prefix
    self.train_sents = np.load(train_sents, allow_pickle=True)
    self.train_labels = np.load(train_labels, allow_pickle=True)
    self.valid_sents = np.load(valid_sents, allow_pickle=True)
    self.valid_labels = np.load(valid_labels, allow_pickle=True)
    self.nlp = English()
    assert len(self.train_sents) == len(self.train_labels)
    assert len(self.valid_sents) == len(self.valid_labels)

  def lowerCase(self):
    self.train_sents = np.array([ts.lower() for ts in self.train_sents])
    self.valid_sents = np.array([vs.lower() for vs in self.valid_sents])

  @staticmethod
  def removeStopWords(sentences, nlp):
    sents = []
    for sent in sentences:
      spacy_obj = nlp(str(sent))
      no_stop = [_.text for _ in spacy_obj if not _.is_stop]
      sents.append(' '.join(no_stop))
    sents = np.array(sents)
    return sents

  @staticmethod
  def removeAlpNum(sentences, nlp):
    sents = []
    for sent in sentences:
      spacy_obj = nlp(str(sent))
      alnum = [_.text for _ in spacy_obj if _.isalnum()]
      sents.append(' '.join(alnum))
    sents = np.array(sents)
    return sents

  def getStats(self, is_resampling=False):
    # if resampling, perform stratified sampling
    if is_resampling:
      from sklearn.model_selection import StratifiedShuffleSplit
      all_sents = np.concatenate((self.train_sents, self.valid_sents))
      all_labels = np.concatenate((self.train_labels, self.valid_labels))
      spliter = StratifiedShuffleSplit(n_splits=1,
                                       test_size=0.2,
                                       random_state=0)
      for train_index, valid_index in spliter.split(all_sents, all_labels):
        print('Stratified training size is {}, validation size is {}'.format(
            len(train_index), len(valid_index)))
        self.train_sents, self.valid_sents = all_sents[train_index], all_sents[
            valid_index]
        self.train_labels, self.valid_labels = all_labels[
            train_index], all_labels[valid_index]
    print('Training set label distribution is {}'.format(
        Counter(self.train_labels)))
    print('Validation set label distribution is {}'.format(
        Counter(self.valid_labels)))

  def listToStr(self):
    # train_sents = [' '.join(ts) for ts in self.train_sents if len(ts) > 1 else ts[0]]
    if self.prefix != 'KW_':
      train_sents = [
          ' '.join(ts) if len(ts) > 1 else ts[0] for ts in self.train_sents
      ]
      valid_sents = [
          ' '.join(vs) if len(vs) > 1 else vs[0] for vs in self.valid_sents
      ]

      self.train_sents = np.array(train_sents)
      self.valid_sents = np.array(valid_sents)

  def onehotEncodingLabel(self, save_dir):
    cats = sorted(
        list(set(np.concatenate((self.train_labels, self.valid_labels)))))
    label_to_int = dict((l, i) for i, l in enumerate(cats))
    print(label_to_int)
    with open(save_dir / 'label_to_int.pkl', 'wb') as f:
      pkl.dump(label_to_int, f)
    int_train_labels = [label_to_int[l] for l in self.train_labels]
    int_valid_labels = [label_to_int[l] for l in self.valid_labels]
    self.onehot_train_labels = []
    for i in int_train_labels:
      temp = np.zeros(len(cats))
      temp[i] = 1
      self.onehot_train_labels.append(temp)
    self.onehot_train_labels = np.array(self.onehot_train_labels)

    self.onehot_valid_labels = []
    for i in int_valid_labels:
      temp = np.zeros(len(cats))
      temp[i] = 1
      self.onehot_valid_labels.append(temp)
    self.onehot_valid_labels = np.array(self.onehot_valid_labels)

  def randomizeData(self, save_dir, prefix):
    # set random seed, affect batch size
    p1 = np.random.permutation(len(self.onehot_train_labels))
    p2 = np.random.permutation(len(self.onehot_valid_labels))
    np.save(save_dir / (prefix + 'permutation_series.npy'), p2)

    self.train_sents = np.array(self.train_sents)[p1]
    self.valid_sents = np.array(self.valid_sents)[p2]
    self.onehot_train_labels = np.array(self.onehot_train_labels)[p1]
    self.onehot_valid_labels = np.array(self.onehot_valid_labels)[p2]

    print("Training sents length is {}, {}".format(len(self.train_sents),
                                                   self.train_sents[:5]))
    print("Training labels length is {}, {}".format(len(self.train_labels),
                                                    self.train_labels[:5]))
    print("Validation sents length is {}, {}".format(len(self.valid_sents),
                                                     self.valid_sents[:5]))
    print("Validation labels length is {}, {}".format(len(self.valid_labels),
                                                      self.valid_labels[:5]))
    # np.save(save_dir / 'input_train_labels.npy', self.onehot_train_labels)
    # np.save(save_dir / 'input_train_sents.npy', self.train_sents)
    # np.save(save_dir / 'input_valid_labels.npy', self.onehot_valid_labels)
    # np.save(save_dir / 'input_valid_sents.npy', self.valid_sents)


class VanillaUSE():

  def __init__(self, train_x, train_y, valid_x, valid_y):
    self.train_x = train_x
    self.train_y = train_y
    self.valid_x = valid_x
    self.valid_y = valid_y
    self.n_labels = self.valid_y.shape[1]
    self.embed = hub.Module('../../../models/use-module', trainable=False)

  def use_embedding(self, x):
    return self.embed(tf.reshape(tf.cast(x, 'string'), [-1]),
                      signature='default',
                      as_dict=True)['default']

  def createModel(self):
    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(self.use_embedding, output_shape=(512,))(input_text)
    dense = Dense(512)(embedding)  #kernel_regularizer=l1(0.0001) #
    dense = BatchNormalization()(dense)
    dense = ReLU()(dense)
    dense = Dropout(0.4)(dense)
    dense = Dense(512)(dense)
    dense = BatchNormalization()(dense)
    dense = ReLU()(dense)
    dense = Dropout(0.4)(dense)
    pred = Dense(self.n_labels, activation='softmax')(dense)
    self.model = Model(inputs=[input_text], outputs=pred)
    # {triangular, triangular2, exp_range}
    self.model.compile(loss='categorical_crossentropy',
                       optimizer="adam",
                       metrics=['accuracy'])
    print(self.model.summary())

  @profile
  def train(self, filepath):
    try:
      os.mkdir(filepath)
    except:
      pass

    def schedule_decay(epoch, lr):
      # exponential decay, init lr: 0.5
      if epoch != 0:
        lr = lr / 2
      return lr

    with tf.Session() as sess:
      K.set_session(sess)
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      #   lr_scheduler = LearningRateScheduler(schedule_decay)
      ckpt = ModelCheckpoint(filepath + '/{epoch:02d}.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')  # ,save_freq="epoch"
      print(self.train_x[:10], self.train_y[:10])
      hist = self.model.fit(self.train_x,
                            self.train_y,
                            validation_split=0.1,
                            epochs=20,
                            batch_size=128,
                            callbacks=[ckpt])
      pred = self.model.predict(self.valid_x)
      self.pred = np.argmax(pred, axis=1)
      self.valid_y_ = np.argmax(self.valid_y, axis=1)

    target_names = ['update', 'new']
    print(accuracy_score(self.valid_y_, self.pred))
    print(accuracy_score(self.valid_y_, self.pred, normalize=False))
    print(precision_recall_fscore_support(self.valid_y_, self.pred))
    print(
        classification_report(self.valid_y_,
                              self.pred,
                              target_names=target_names))
    print(confusion_matrix(self.valid_y_, self.pred))

  def consolidateResult(self, filepath):
    import pandas as pd
    df = pd.DataFrame(list(zip(self.pred, self.valid_y_)),
                      columns=['Pred', 'GroundTruth'])
    df.to_csv(filepath + '/result.csv')

    print()


if __name__ == '__main__':
  start_time = time.time()
  # exprmt_dir = 'pairwise_exprmt/data_v0/'
  # serial_no = '13-0/'
  # input_path = Path(exprmt_dir + serial_no)
  # prefix = ''  # 'KW_' | 'rev_scp' | 'scp'

  parent_dir = './processed_data/'
  serial_no = 'split_0/'
  prefix = 'rev_scp_'  # 'KW_' | 'rev_scp_' | 'scp_'
  save_dir = parent_dir + serial_no
  input_path = Path(parent_dir + serial_no)

  train_sents_path = input_path / (prefix + 'train_email.npy')
  train_labels_path = input_path / (prefix + 'train_label.npy')
  valid_sents_path = input_path / (prefix + 'valid_email.npy')
  valid_labels_path = input_path / (prefix + 'valid_label.npy')

  dr = ModelDataReader(prefix, train_sents_path, train_labels_path,
                       valid_sents_path, valid_labels_path)
  dr.getStats()
  dr.onehotEncodingLabel(input_path)
  dr.listToStr()
  dr.randomizeData(input_path)

  vu = VanillaUSE(dr.train_sents, dr.onehot_train_labels, dr.valid_sents,
                  dr.onehot_valid_labels)
  vu.createModel()
  vu.train(filepath=str(input_path) + '/' + prefix + 'model')
  vu.consolidateResult(str(input_path) + '/' + prefix + 'model')
  print('Overall Time: ', str(time.time() - start_time), 's')
