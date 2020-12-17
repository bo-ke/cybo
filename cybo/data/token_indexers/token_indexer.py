# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: token_indexer.py
@time: 2020/12/16 23:27:54

这一行开始写关于本文件的说明与解释


'''
from typing import Dict, List, Any

from cybo.data.tokenizers.token import Token

IndexedTokenList = Dict[str, List[Any]]


class TokenIndexer():
    def __init__(self):
        pass

    def count_vocab_items(self, tokens: Token, counter: Dict[str, int]):
        raise NotImplementedError

    def tokens_to_indices(self, tokens: List[Token], vocab):
        raise NotImplementedError

    def indices_to_tokens(self, indexed_tokens):
        raise NotImplementedError
