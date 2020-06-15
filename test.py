
"""
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