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
