# know data - 20200709
import numpy as np
import pandas as pd
# data_dir 'processing_pipeline_exprmt_20200610/split_2/'

if __name__ == '__main__':

  train_v1 = np.load('KW_train_email.npy', allow_pickle=True)
  train_v2 = np.load('rev_scp_train_email.npy', allow_pickle=True)
  train_v0 = np.load('scp_keep_long_train_email.npy', allow_pickle=True)
  train_label = np.load('KW_train_label.npy', allow_pickle=True)

  valid_v1 = np.load('KW_valid_email.npy', allow_pickle=True)
  valid_v2 = np.load('rev_scp_valid_email.npy', allow_pickle=True)
  valid_v0 = np.load('scp_keep_long_valid_email.npy', allow_pickle=True)
  valid_label = np.load('KW_valid_label.npy', allow_pickle=True)

  df_train = pd.DataFrame(
      columns=['v0_sents', 'v1_sents', 'v2_sents', 'labels'])
  df_train.v1_sents = train_v1
  df_train.v2_sents = train_v2
  df_train.v0_sents = train_v0
  df_train.labels = train_label
  df_train.to_csv('./train_v012.csv')

  df_valid = pd.DataFrame(
      columns=['v0_sents', 'v1_sents', 'v2_sents', 'labels'])
  df_valid.v0_sents = valid_v0
  df_valid.v1_sents = valid_v1
  df_valid.v2_sents = valid_v2
  df_valid.labels = valid_label
  df_valid.to_csv('./valid_v012.csv')
