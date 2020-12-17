# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: single_id_token_indexer.py
@time: 2020/12/17 23:43:19

这一行开始写关于本文件的说明与解释


'''
from typing import List, Optional
from overrides import overrides

from cybo.data.tokenizers.token import Token
from cybo.data.token_indexers.token_indexer import TokenIndexer


class SingleIdTokenIndexer(TokenIndexer):
    def __init__(self,
                 namespace:  Optional[str] = "tokens",
                 lowercase_tokens: bool = False,
                 feature_name: str = "text"):
        """
        Args:
            feature_name (str, optional): `Tokens` features Defaults to "text".
        """
        self.lowercase_tokens = lowercase_tokens
        self.feature_name = feature_name
        super().__init__()

    @overrides
    def count_vocab_items(self, tokens, counter):
        return super().count_vocab_items(tokens, counter)

    @overrides
    def tokens_to_indices(self, tokens: List[Token], vocab):
        indices = List[int] = []

        for token in tokens:
            text = self._get_feature_value(token)
            if self.lowercase_tokens:
                text = text.lower()
            indices.append()
            # todo


    def _get_feature_value(self, token: Token) -> str:
        """
        获取Token feature_name 对应的value
        """
        text = getattr(token, self.feature_name)
        return text
