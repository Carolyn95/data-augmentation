"""
use 0603 envr
ssh files under processed_data and data
{'unknow': 0, 'update': 1, 'new': 2}
"""

from collections import Counter
import re
import os
import pdb
import numpy as np
import pickle as pkl
import tensorflow as tf
import keras.backend as K
import tensorflow_hub as hub
from keras.models import Model
from keras.regularizers import l1
from keras.layers import Input, Lambda, Dense, Dropout, Reshape, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
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

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove logging info
import warnings
warnings.filterwarnings('ignore')  # filter out warnings


class DataReader():

  def __init__(self, train_sents, train_labels, valid_sents, valid_labels):
    self.train_sents = np.load(train_sents, allow_pickle=True)
    self.train_labels = np.load(train_labels, allow_pickle=True)
    self.valid_sents = np.load(valid_sents, allow_pickle=True)
    self.valid_labels = np.load(valid_labels, allow_pickle=True)

    assert len(self.train_sents) == len(self.train_labels)
    assert len(self.valid_sents) == len(self.valid_labels)

  def lowerCase(self):
    self.train_sents = np.array([ts.lower() for ts in self.train_sents])
    self.valid_sents = np.array([vs.lower() for vs in self.valid_sents])

  def randomizeData(self):
    p1 = np.random.permutation(len(self.train_labels))
    p2 = np.random.permutation(len(self.valid_labels))

    self.train_sents = np.array(self.train_sents)[p1]
    self.valid_sents = np.array(self.valid_sents)[p2]
    self.train_labels = np.array(self.train_labels)[p1]
    self.valid_labels = np.array(self.valid_labels)[p2]

    print(len(self.train_sents), len(self.valid_sents), len(self.train_labels),
          len(self.valid_labels))


class VanillaUSE():

  def __init__(self, train_x, train_y, valid_x, valid_y):
    self.train_x = train_x
    self.train_y = train_y
    self.valid_x = valid_x
    self.valid_y = valid_y
    self.n_labels = self.valid_y.shape[1]
    self.embed = hub.Module('../models/use-module', trainable=False)

  def use_embedding(self, x):
    return self.embed(tf.reshape(tf.cast(x, 'string'), [-1]),
                      signature='default',
                      as_dict=True)['default']

  def createModel(self):
    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(self.use_embedding, output_shape=(512,))(input_text)
    dense = Dense(512, activation='relu')(
        embedding)  #kernel_regularizer=l1(0.0001) #
    dense = Dense(512, activation='tanh')(dense)
    dense = Dense(256, activation='relu')(dense)
    dense = Dropout(0.4)(dense)
    pred = Dense(self.n_labels, activation='softmax')(dense)
    self.model = Model(inputs=[input_text], outputs=pred)
    self.model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
    print(self.model.summary())

  def createModelBN(self):
    # create model with batch normalization layers
    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(self.use_embedding, output_shape=(512,))(input_text)
    dense = Dense(256, activation='relu',
                  kernel_regularizer=l1(0.0001))(embedding)
    dense = BatchNormalization()(dense)
    dense = Dense(256, activation='tanh')(dense)
    # dense = Dropout(0.1)(dense)
    dense = BatchNormalization()(dense)
    pred = Dense(self.n_labels, activation='softmax')(dense)
    self.model = Model(inputs=[input_text], outputs=pred)
    self.model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
    # opt = Adam(learning_rate=0.001)  # default is 0.001
    # self.model.compile(loss='categorical_crossentropy',
    #                    optimizer=opt,
    #                    metrics=['accuracy'])
    print(self.model.summary())

  @profile
  def train(self, filepath):
    try:
      os.mkdir(filepath)
    except:
      pass
    with tf.Session() as sess:
      K.set_session(sess)
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      ckpt = ModelCheckpoint(filepath + '/{epoch:02d}.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')
      hist = self.model.fit(self.train_x,
                            self.train_y,
                            validation_split=0.2,
                            epochs=20,
                            batch_size=128,
                            callbacks=[ckpt])
      pred = self.model.predict(self.valid_x)
      self.pred = np.argmax(pred, axis=1)
      self.valid_y_ = np.argmax(self.valid_y, axis=1)

    target_names = ['unknow', 'update', 'new']
    print(accuracy_score(self.valid_y_, self.pred))
    print(accuracy_score(self.valid_y_, self.pred, normalize=False))
    print(precision_recall_fscore_support(self.valid_y_, self.pred))
    print(
        classification_report(self.valid_y_,
                              self.pred,
                              target_names=target_names))

  def consolidateResult(self, filepath):
    import pandas as pd
    df = pd.DataFrame(list(zip(self.pred, self.valid_y_)),
                      columns=['Pred', 'GroundTruth'])
    df.to_csv(filepath + '/result.csv')

    print()


if __name__ == '__main__':
  start_time = time.time()
  # input data: "processed_data/randomized_sents_1k.npy" | "processed_data/randomized_sents_2k.npy"
  # input label: "processed_data/randomized_labels_1k.npy" | "processed_data/randomized_labels_2k.npy"
  # train_sents_path = 'processed_data/randomized_sents_1k.npy'
  # train_labels_path = 'processed_data/randomized_labels_1k.npy'
  train_sents_path = 'processed_data/randomized_sents_2k.npy'
  train_labels_path = 'processed_data/randomized_labels_2k.npy'
  valid_sents_path = 'data/valid_sents_mixed.npy'
  valid_labels_path = 'data/valid_labels_onehot_mixed.npy'
  dr = DataReader(train_sents_path, train_labels_path, valid_sents_path,
                  valid_labels_path)
  # pdb.set_trace()
  # dr.lowerCase()
  dr.randomizeData()

  vu = VanillaUSE(dr.train_sents, dr.train_labels, dr.valid_sents,
                  dr.valid_labels)
  vu.createModel()
  vu.train(filepath='serial-no/20')
  vu.consolidateResult(filepath='serial-no/20')
  # vu.createModelBN()
  # vu.train(filepath='Vanilla_USE_BN')
  # vu.consolidateResult(filepath='Vanilla_USE_BN')
  print('Overall Time: ', str(time.time() - start_time), 's')
