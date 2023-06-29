# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import math
import collections
import pickle
import os
from random import shuffle

import numpy as np
from transformers import PreTrainedTokenizerBase


class DataCollatorForT5DocCLS:
    """
    Data collator used for T5 document classification
    """
    def __init__(self, json_data=None, tokenizer=None, meta_path=None, input_length=None, target_length=None, pad_token_id=None, decoder_start_token_id=None):

        self.json_data = json_data
        self.tokenizer = tokenizer #이전에 만든 udop tokenizer를 불러옴
        self.input_length = input_length
        self.target_length = target_length
        self.pad_token_id = pad_token_id
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, user_prompt ,ori_input_ids, ori_bbox_list, labels=None):

        # "원래 input text 정보 & bounding box"
        # -->
        # "prompt text 정보 + 원래 input text 정보" list
        # +
        # [0,0,0,0]을 promt text token 개수만큼 + 원래 bounding box

        #prompt_text = 'document classification.'
        prompt_text = user_prompt
        prompt_ids =  self.tokenizer.encode(prompt_text, add_special_tokens=False)
        input_ids = prompt_ids + ori_input_ids
        bbox_list = [[0,0,0,0]] * len(prompt_ids) + ori_bbox_list

        if(labels!=None):  
          labels = self.tokenizer.encode(labels, add_special_tokens=True)

        return input_ids, labels, bbox_list
