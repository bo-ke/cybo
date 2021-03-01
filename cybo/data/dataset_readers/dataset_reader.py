# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: dataset_reader.py
@time: 2020/12/16 22:25:09

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from collections import Iterator
from pydantic import BaseModel
from typing import List, Dict

from cybo.common.checks import ConfigurationError
from cybo.data.instance import Instance
from cybo.data.vocabulary import Vocabulary


class InputExample(BaseModel):
    guid: int
    text: List[str]
    label: str

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for token in self.text:
            counter["text"][token] += 1
        counter["label"][self.label] += 1


class InputFeatures(BaseModel):
    input_ids: List[int]
    label: List[int]

    @classmethod
    def return_types(cls):
        return {"input_ids": tf.int32, "label": tf.int32}


class DatasetReader():
    """
    dataset_reader 基类
    """

    def __init__(self, tokenizer, token_indexer):
        self._tokenizer = tokenizer
        self._token_indexer = token_indexer

    def read(self, file_path: str) -> List[Instance]:
        text_iterator = self.read_file(file_path=file_path)
        instances = [self.text_to_instance(text) for text in text_iterator]
        if not instances:
            raise ConfigurationError(
                "No instances were read from the given filepath {}. "
                "Is the path correct?".format(file_path))
        return instances

    def read_file(self, file_path: str) -> Iterator:
        raise NotImplementedError

    def text_to_instance(self, inputs) -> Instance:
        raise NotImplementedError

    def convert_instances_to_features(
            self, instances: List[Instance],
            vocab: Vocabulary) -> List[InputFeatures]:
        raise NotImplementedError

    def encode_plus(self, text) -> InputFeatures:
        raise NotImplementedError


class Tokenizer():
    def tokenize(self, text):
        text = text.strip()
        if not text:
            return []
        tokens = list(text)
        return tokens
