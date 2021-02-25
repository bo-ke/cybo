# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: text_classifier_dataset_reader.py
@time: 2020/12/16 22:29:04

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from typing import Dict, List, Optional
import pandas as pd
from collections import Iterator

from cybo.data.dataset_readers.dataset_reader import DatasetReader, InputFeatures
from cybo.data.fields.text_field import TextField
from cybo.data.fields.field import Field
from cybo.data.instance import Instance
from cybo.data.vocabulary import Vocabulary


class TextClassifierInputFeatures(InputFeatures):
    input_ids: List[int]
    label_ids: Optional[List[int]]


class TextClassifierDatasetReader(DatasetReader):
    """
    文本分类 dataset_reader
    用于读取 text, label
    (csv格式文件)
    """

    def __init__(self, tokenizer, token_indexer):
        super().__init__(tokenizer, token_indexer)

    def read_file(self, file_path) -> Iterator:
        df = pd.read_csv(file_path, encoding="utf-8")
        for text, label in zip(df["text"], df["label"]):
            yield text, label

    def text_to_instance(self, inputs):
        fields: Dict[str, Field] = {}
        text, label = inputs
        tokens = self._tokenizer(text)
        fields["tokens"] = TextField(tokens, self._token_indexer)

        return Instance(fields)

    def _truncated(self, tokens):
        if len(tokens) > self._max_seq_len:
            tokens = tokens[:self._max_seq_len]
        return tokens

    def convert_instances_to_features(
            self, instances: List[Instance],
            vocab: Vocabulary) -> List[TextClassifierInputFeatures]:
        features = []
        for instance in instances:
            instance.index_fields(vocab=vocab)

            input_ids = instance["tokens"].indexd_tokens
            print(input_ids)
