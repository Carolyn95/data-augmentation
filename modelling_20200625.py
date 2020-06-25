# pairwise comparison

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

    # def schedule_decay(epoch, lr):
    #   # linear decay, init lr: 0.25
    #   if epoch != 0:
    #     lr = lr - 0.01
    #   return lr

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
                             save_best_only=False,
                             mode='auto',
                             save_freq="epoch")  # ,save_freq="epoch"
      hist = self.model.fit(self.train_x,
                            self.train_y,
                            validation_split=0.2,
                            epochs=20,
                            batch_size=128,
                            callbacks=[ckpt])
      # print(hist)
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
  train_sents_path = 'processed_data/train_sents_pairwise_update_new.npy'
  train_labels_path = 'processed_data/train_labels_pairwise_update_new.npy'
  valid_sents_path = 'processed_data/valid_sents_pairwise_update_new.npy'
  valid_labels_path = 'processed_data/valid_labels_pairwise_update_new.npy'
  dr = DataReader(train_sents_path, train_labels_path, valid_sents_path,
                  valid_labels_path)
  dr.randomizeData()

  vu = VanillaUSE(dr.train_sents, dr.train_labels, dr.valid_sents,
                  dr.valid_labels)
  vu.createModel()
  vu.train(filepath='serial-no/13')
  vu.consolidateResult(filepath='serial-no/13')
  print('Overall Time: ', str(time.time() - start_time), 's')
