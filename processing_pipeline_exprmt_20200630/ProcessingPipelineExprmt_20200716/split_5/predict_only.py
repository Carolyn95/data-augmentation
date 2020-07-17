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
import numpy as np
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.metrics import confusion_matrix
import pandas as pd 

# not working, if reorder dataset, result changed dramatically


embed = hub.Module('../../../../../models/use-module', trainable=False)

def use_embedding(x):
  return embed(tf.reshape(tf.cast(x, 'string'), [-1]),
                    signature='default',
                    as_dict=True)['default']

def createModel(n_labels):
  input_text = Input(shape=(1,), dtype='string')
  embedding = Lambda(use_embedding, output_shape=(512,))(input_text)
  dense = Dense(512)(embedding)  #kernel_regularizer=l1(0.0001) #
  dense = BatchNormalization()(dense)
  dense = ReLU()(dense)
  dense = Dropout(0.4)(dense)
  dense = Dense(512)(dense)
  dense = BatchNormalization()(dense)
  dense = ReLU()(dense)
  dense = Dropout(0.4)(dense)
  pred = Dense(n_labels, activation='softmax')(dense)
  model = Model(inputs=[input_text], outputs=pred)
  # {triangular, triangular2, exp_range}
  model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])
  print(model.summary())
  return model

if __name__ == '__main__':
  model_weights = '../split_4/scp_model/20.hdf5'
  sents = np.load('./scp_json_emails.npy', allow_pickle=True)
  # sents = sents[np.random.permutation(len(sents))]
  sents = np.array([' '.join(s) if len(s) > 1 else s[0] for s in sents])
  actual_labels = np.load('./scp_json_labels.npy', allow_pickle=True)
  print(Counter(actual_labels))
  mapping = {0: 'new', 1: 'update'}
  with tf.Session() as sess:
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    model = createModel(2)
    model.load_weights(model_weights)
    pred = model.predict(sents).argmax(axis=1)
    pred_cat = [mapping[i] for i in pred]
    print(accuracy_score(actual_labels, pred_cat))
    print(confusion_matrix(actual_labels, pred_cat))
    df = pd.DataFrame(columns=['Sents', 'Pred', 'Actual'])
    df.Sents = sents 
    df.Pred = pred_cat
    df.Actual = actual_labels 
    df.to_csv('./scp_json.csv')

