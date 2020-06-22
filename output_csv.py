# _*_ coding: utf-8 _*_
# __author__='Carolyn CHEN'
'''
This script is for 
reading original data from json format
saving them as csv
'''

import pandas as pd
import json
import os
from bs4 import BeautifulSoup
import pdb


def main(file_path):
  columns = ['EmailID', 'EmailContent']
  df = pd.DataFrame(columns=columns)
  for idx, file_name in enumerate(os.listdir(file_path)):
    json_file = file_path + '/' + file_name
    print('Now is processing -', idx, '-', file_name)
    with open(os.path.normpath(json_file)) as jf:
      email = json.load(jf)
      id = email['EmailID']
      body = email['EmailBody']
      soup = BeautifulSoup(body, features="html.parser")
      email_body_text = soup.get_text()
      df.loc[idx, 'EmailID'] = id
      df.loc[idx, 'EmailContent'] = email_body_text
  print('main function ends, now saving csv')
  df.to_csv(file_path + '.csv')


if __name__ == '__main__':
  file_path = "ntuc-unzip/"
  label_1 = "NEW"
  label_2 = "RESOLVED"
  label_3 = "UNKNOWN"
  label_4 = "UPDATE"

  FILE_PATH = file_path + label_1
  main(FILE_PATH)
