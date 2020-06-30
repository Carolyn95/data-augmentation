"""
Master piece for:
1. Convert Excel to Array
2. Convert Json to Array
"""
import pdb
import pandas as pd
from pathlib import Path
import numpy as np
from rich import print
import os
import json


class ExcelToArray():

  def __init__(self, path, sheet_name):
    self.df = pd.read_excel(path, sheet_name=sheet_name)

  def parseColumns(self, save_dir, prefix=None):
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    self.bodies = np.array(self.df.Body)
    self.intentions = np.array(self.df.Categories)
    np.save(save_dir + '/' + prefix + 'bodies.npy', self.bodies)
    np.save(save_dir + '/' + prefix + 'intentions.npy', self.intentions)
    print(len(self.bodies), len(self.intentions))


class JsonToArray():

  def __init__(self, path):
    with open(os.path.normpath(path), 'r') as f:
      self.data = f.read()
    self.data = self.data.splitlines()
    self.emails = []
    self.labels = []

  def parseJson(self):
    for i, d in enumerate(self.data):
      d = eval(d)
      temp = d['content'].copy()
      temp = ''.join(temp)
      self.emails.append(temp)
      self.labels.append(d['intent'])

  def saveAsArray(self, save_path, prefix=None):
    if not os.path.exists(save_path):
      os.mkdir(save_path)
    self.emails = np.array(self.emails)
    self.labels = np.array(self.labels)

    print(len(self.emails), len(self.labels))
    np.save(save_path + prefix + 'bodies.npy', self.emails)
    np.save(save_path + prefix + 'intentions.npy', self.labels)


if __name__ == '__main__':
  eta_dec = ExcelToArray(path='org_data/SME_emails.xlsx',
                         sheet_name='NTUCDec2019')
  eta_dec.parseColumns(save_dir='org_array_data/', prefix='dec_')
  eta_jan = ExcelToArray(path='org_data/SME_emails.xlsx',
                         sheet_name='NTUC-Jan-2020')
  eta_jan.parseColumns(save_dir='org_array_data/', prefix='jan_')

  train_json = JsonToArray(path='org_data/ntuc_train.json')
  train_json.parseJson()
  train_json.saveAsArray(save_path='org_array_data/', prefix='json_train_')

  valid_json = JsonToArray(path='org_data/ntuc_valid.json')
  valid_json.parseJson()
  valid_json.saveAsArray(save_path='org_array_data/', prefix='json_valid_')
