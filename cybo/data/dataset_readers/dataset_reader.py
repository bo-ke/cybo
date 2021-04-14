# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: dataset_reader.py
@time: 2020/12/16 22:25:09

这一行开始写关于本文件的说明与解释


'''
import json
import tensorflow as tf
from collections import Iterator
from pydantic import BaseModel
from typing import List, Dict

from cybo.common.logger import logger
from cybo.common.checks import ConfigurationError
from cybo.data.vocabulary import Vocabulary
from cybo.data.tokenizers import Tokenizer


class InputExample(BaseModel):
    guid: int

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        # for token in self.text:
        #     counter["text"][token] += 1
        # counter["label"][self.label] += 1
        raise NotImplementedError


class InputFeatures(BaseModel):
    input_ids: List[int]

    @classmethod
    def return_types(cls):
        # return {"input_ids": tf.int32, "label": tf.int32}
        raise NotImplementedError


class DatasetReader():
    def __init__(self, tokenizer: Tokenizer = None) -> None:
        self._tokenizer = tokenizer

    def get_examples(self, filepath) -> List[InputExample]:
        raise NotImplementedError

    @classmethod
    def read_file(cls, filepath):
        lines = open(filepath, "r", encoding="utf-8").readlines()
        return lines

    def convert_examples_to_features(
            self, examples: List[InputExample],
            vocab: Vocabulary, max_seq_length: int = 32,
            verbose: bool = False):
        features = []
        for (ex_index, example) in enumerate(examples):
            features_item = self._convert_example_to_features(
                example, vocab, max_seq_length)
            if verbose:
                if ex_index <= 5:
                    print("*** Example ** *")
                    for k, v in example.dict().items():
                        print(f"{k}: {v}")
                    print("*** Features ** *")
                    for k, v in features_item.dict().items():
                        print(f"{k}: {v}")
            features.append(features_item)
        return features

    def _convert_example_to_features(
            self, example: InputExample,
            vocab: Vocabulary, max_seq_length: int = 32) -> InputFeatures:
        raise NotImplementedError

    def encode_plus(self, text, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @classmethod
    def _truncated_add_padded(cls, tokens, max_seq_length, padding_token=0):
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        else:
            tokens = tokens + [padding_token] * (max_seq_length - len(tokens))
        return tokens

    @property
    def return_types(self):
        # todo
        # 用于dataloader from generator 手动传入return_types
        return InputFeatures.return_types()
