# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: text_field.py
@time: 2020/12/16 23:07:19

这一行开始写关于本文件的说明与解释


'''
from typing import Dict, List
import tensorflow as tf
from overrides import overrides

from cybo.data.fields.sequence_field import SequenceField
from cybo.data.tokenizers.token import Token
from cybo.data.token_indexers.token_indexer import TokenIndexer

TextFieldTensors = Dict[str, Dict[str, tf.Tensor]]


class TextField(SequenceField[TextFieldTensors]):
    def __init__(
            self, tokens: List[Token],
            token_indexer: TokenIndexer) -> None:
        self.tokens = tokens
        self._token_indexer = token_indexer
        self.indexed_tokens = None

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for token in self.tokens:
            self._token_indexer.count_vocab_items(token, counter)

    @overrides
    def index(self, vocab):
        self.indexed_tokens = self._token_indexer.tokens_to_indices(
            self.tokens, vocab)
