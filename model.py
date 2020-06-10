from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange

import torch
import torch.nn.functional as F
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

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
# ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys())
#                   for conf in (GPT2Config, OpenAIGPTConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
}


def set_seed(args):
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits,
                          top_k=0,
                          top_p=0.0,
                          filter_value=-float('Inf'),
                          stop_token_id=None):
  """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """

  if stop_token_id:
    # logits[:, stop_token_id] = filter_value
    logits[:, stop_token_id] = filter_value
  top_k = min(top_k, logits.size(-1))  # Safety check
  if top_k > 0:
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = filter_value

  if top_p > 0.0:
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens with cumulative probability above the threshold
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[...,
                             1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        dim=1, index=sorted_indices, src=sorted_indices_to_remove)
    logits[indices_to_remove] = filter_value
  return logits


def sample_sequence(model,
                    length,
                    context,
                    num_samples=1,
                    temperature=1,
                    top_k=0,
                    top_p=0.0,
                    repetition_penalty=1.0,
                    is_xlnet=False,
                    is_xlm_mlm=False,
                    xlm_mask_token=None,
                    xlm_lang=None,
                    device='cpu',
                    stop_token_ids=None):
  context = torch.tensor(context, dtype=torch.long, device=device)
  context = context.unsqueeze(0).repeat(num_samples, 1)
  generated = context
  with torch.no_grad():
    for _ in trange(length):

      inputs = {'input_ids': generated}
      if is_xlnet:
        # XLNet is a direct (predict same token, not next token) and bi-directional model by default
        # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
        input_ids = torch.cat(
            (generated, torch.zeros((1, 1), dtype=torch.long, device=device)),
            dim=1)
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]),
                                dtype=torch.float,
                                device=device)
        perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
        target_mapping = torch.zeros((1, 1, input_ids.shape[1]),
                                     dtype=torch.float,
                                     device=device)
        target_mapping[0, 0, -1] = 1.0  # predict last token
        inputs = {
            'input_ids': input_ids,
            'perm_mask': perm_mask,
            'target_mapping': target_mapping
        }

      if is_xlm_mlm and xlm_mask_token:
        # XLM MLM models are direct models (predict same token, not next token)
        # => need one additional dummy token in the input (will be masked and guessed)
        input_ids = torch.cat(
            (generated,
             torch.full(
                 (1, 1), xlm_mask_token, dtype=torch.long, device=device)),
            dim=1)
        inputs = {'input_ids': input_ids}

      if xlm_lang is not None:
        inputs["langs"] = torch.tensor([xlm_lang] *
                                       inputs["input_ids"].shape[1],
                                       device=device).view(1, -1)

      outputs = model(
          **inputs
      )  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
      next_token_logits = outputs[0][:, -1, :] / (temperature
                                                  if temperature > 0 else 1.)

      # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
      for i in range(num_samples):
        for _ in set(generated[i].tolist()):
          next_token_logits[i, _] /= repetition_penalty

      filtered_logits = top_k_top_p_filtering(next_token_logits,
                                              top_k=top_k,
                                              top_p=top_p,
                                              stop_token_id=stop_token_ids)

      if temperature == 0:  # greedy sampling:
        next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
      else:
        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1),
                                       num_samples=1)
      generated = torch.cat((generated, next_token), dim=1)
  return generated


params = {
    'model_type': 'gpt2',
    'model_name_or_path': '',
    'prompt': '',
    'padding_text': '',
    'xlm_lang': '',
    'length': 20,
    'num_samples': 10,
    'temperature': 1.0,
    'repetition_penalty': 1.0,
    'top_k': 20,
    'top_p': 0.0,
    'no_cuda': False,
    'seed': 42,
    'stop_token': None
}

from input_reader import Namespace
args = Namespace()

for key, val in params.items():
  args.add(key, val)

args.device = torch.device(
    "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = torch.cuda.device_count()

set_seed(args)

args.model_type = args.model_type.lower()


class ParaphraseModel():

  def __init__(self, model_path):

    self.model_path = model_path


#         self.load_model(model_path)

  def load_model(self, model_path):

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    self.tokenizer = tokenizer_class.from_pretrained(model_path)
    self.model = model_class.from_pretrained(model_path)
    a = self.model.to(args.device)

    if args.length < 0 and self.model.config.max_position_embeddings > 0:
      args.length = self.model.config.max_position_embeddings
    elif 0 < self.model.config.max_position_embeddings < args.length:
      args.length = self.model.config.max_position_embeddings  # No generation bigger than model size
    elif args.length < 0:
      args.length = MAX_LENGTH  # avoid infinite loop

  def get_paraphrases(self, prompt, num_samples, stop_words):

    set_seed(args)
    self.load_model(self.model_path)
    prompt += ' >>>>>>>>'
    prompt = prompt.lower()

    stop_words = stop_words.split(";")
    stop_words = [word.strip() for word in stop_words]
    added_stop_words = ['Ġ' + word for word in stop_words if word]

    all_stop_words = stop_words + added_stop_words

    raw_text = prompt

    all_preds = []
    if args.model_type in ["transfo-xl", "xlnet"]:
      # Models with memory likes to have a long prompt for short inputs.
      raw_text = (args.padding_text
                  if args.padding_text else PADDING_TEXT) + raw_text
    context_tokens = self.tokenizer.encode(raw_text, add_special_tokens=False)

    stop_token_ids = []
    for word in all_stop_words:
      token_ids = self.tokenizer.convert_tokens_to_ids([word])

      if 50256 in token_ids:
        token_ids.remove(50256)
      stop_token_ids += token_ids

    if args.model_type == "ctrl":
      if not any(context_tokens[0] == x
                 for x in self.tokenizer.control_codes.values()):
        logger.info(
            "WARNING! You are not starting your generation from a control code so you won't get good results"
        )
    out = sample_sequence(
        model=self.model,
        context=context_tokens,
        num_samples=num_samples,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=args.device,
        stop_token_ids=stop_token_ids,
    )
    out = out[:, len(context_tokens):].tolist()
    for o in out:

      text = self.tokenizer.decode(o, clean_up_tokenization_spaces=True)
      text = text[:text.find(args.stop_token) if args.stop_token else None]
      all_preds.append(text.split('<|endoftext|>')[0].strip())

    return all_preds
