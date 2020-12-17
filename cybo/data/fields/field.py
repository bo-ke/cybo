# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: field.py
@time: 2020/12/16 22:57:52

这一行开始写关于本文件的说明与解释


'''
from typing import Dict, Generic, List, TypeVar
from copy import deepcopy

import tensorflow as tf

from cybo.data.vocabulary import Vocabulary

DataArray = TypeVar("DataArray", tf.Tensor, Dict[str, tf.Tensor])


class Field(Generic[DataArray]):
    __slots__ = []  # type: ignore

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        pass

    def index(self, vocab: Vocabulary):
        pass

    def get_padding_lengths(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def duplicate(self):
        return deepcopy(self)
