"""
# 3 entries
# data reader
# data processor
# classifier

# ensure same split in different processing pipeline are using the same set of data
# the only difference among group 13 | 14 | 15 is processing pipeline
# the difference among different splits in one group is the split
"""

from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
import pdb


class DataReader():
  # create split here
  # save into different folders
  def __init__(self, data_dir, save_dir, **kwargs):
    print()

  def removeEmpty(self):
    print()

  def randomSplitOrgData(self):
    print()

  def stratifiedSplitOrgData(self):
    print()


class DataProcessor():
  # different processing pipeline here
  def __init__(self, data_dir):
    print()

  def cleanBySigClf(self):
    print()

  def cleanByKW(self):
    print()

  def cleanBySigClfRev():
    print()


class Classifier():
  # moddeling
  def __init__(self):
    print()


if __name__ == '__main__':
  save_dir = 'processing_pipeline_exprmt_20200630'
  serial_no = 'split_1'
  data_dir = 'all_org'