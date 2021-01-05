# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: text_classifier_dataset_reader.py
@time: 2020/12/16 22:29:04

这一行开始写关于本文件的说明与解释


'''
from typing import Dict
import pandas as pd
from collections import Iterator

from cybo.data.dataset_readers.dataset_reader import DatasetReader
from cybo.data.fields.text_field import TextField


class TextClassifierDatasetReader(DatasetReader):
    """
    文本分类 dataset_reader
    用于读取 text, label  
    (csv格式文件)
    """

    def __init__(self, tokenizer, token_indexer, max_seq_len):
        super().__init__(tokenizer, token_indexer, max_seq_len)

    def read_file(self, file_path) -> Iterator:
        df = pd.read_csv(file_path, encoding="utf-8")
        for text, label in zip(df["text"], df["label"]):
            yield text, label

    def text_to_instance(self, inputs):
        fields: Dict[str, Field] = {}
        text, label = inputs
        tokens = self._tokenizer(text)
        if self._max_seq_len:
            self._truncated(tokens)
        fields["tokens"] = TextField(tokens, self._token_indexer)

        return {"text": text}

    def _truncated(self, tokens):
        if len(tokens) > self._max_seq_len:
            tokens = tokens[:self._max_seq_len]
        return tokens