import ipywidgets as widgets
import requests, os
from IPython.display import display
from ipywidgets import interact
import tqdm

from rasa.nlu.training_data import TrainingData, Message

### Download model

#taken from this StackOverflow answer: https://stackoverflow.com/a/39225039


def download_file_from_google_drive(id, destination):
  URL = "https://docs.google.com/uc?export=download"

  session = requests.Session()

  response = session.get(URL, params={'id': id}, stream=True)
  token = get_confirm_token(response)

  if token:
    params = {'id': id, 'confirm': token}
    response = session.get(URL, params=params, stream=True)

  save_response_content(response, destination)


def get_confirm_token(response):
  for key, value in response.cookies.items():
    if key.startswith('download_warning'):
      return value

  return None


def save_response_content(response, destination):
  CHUNK_SIZE = 32768

  with open(destination, "wb") as f:
    for chunk in response.iter_content(CHUNK_SIZE):
      if chunk:  # filter out keep-alive new chunks
        f.write(chunk)


model_class_file_id = '1N1kn2b7i2ND7eNefzyJM-k13IM8tqZvr'
checkpoint_file_id = '1G0nwXlvzGsb8Ar-OAnYBQKFvY97WMzBy'
model_class_destination = 'model.py'
checkpoint_destination = 'model.zip'
checkpoint_unzipped_destination = 'package_models'
# download models & model module
if not os.path.exists(checkpoint_unzipped_destination):
  tqdm(
      download_file_from_google_drive(checkpoint_file_id,
                                      checkpoint_destination))

import zipfile
import os
with zipfile.ZipFile(checkpoint_destination, 'r') as zip_ref:
  zip_ref.extractall(os.getcwd())

if not os.path.exists(model_class_destination):
  tqdm(
      download_file_from_google_drive(model_class_file_id,
                                      model_class_destination))


