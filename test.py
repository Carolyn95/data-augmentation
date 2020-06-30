""" 20200628
import numpy as np
train_labels = np.load('processed_data/all_onthot_labels_balanced.npy')
valid_labels = np.load('data/valid_labels_onehot_mixed.npy')

int_train_labels = np.argmax(train_labels, axis=1)
int_valid_labels = np.argmax(valid_labels, axis=1)

mapping = {0: 'unknown', 1: 'update', 2: 'new'}

str_train_labels = [mapping[i] for i in int_train_labels]
str_valid_labels = [mapping[i] for i in int_valid_labels]

np.save('processed_data/all_str_labels_balanced.npy', str_train_labels)
np.save('data/valid_labels_str_mixed.npy', str_valid_labels)
"""
""" bef20200628
from transformers import GPT2Config, OpenAIGPTConfig
import pdb

# ALL_MODELS = sum((tuple(GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP.keys()), ()))

GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "gpt2":
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-config.json",
    "gpt2-medium":
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-config.json",
    "gpt2-large":
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-config.json",
    "gpt2-xl":
        "https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-xl-config.json",
    "distilgpt2":
        "https://s3.amazonaws.com/models.huggingface.co/bert/distilgpt2-config.json",
}

OPENAI_GPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai-gpt":
        "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-config.json"
}

ALL_MODELS = tuple(GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP.keys())
# for conf in (GPT2Config, OpenAIGPTConfig):
#   pdb.set_trace()
#   print(type(conf))
"""