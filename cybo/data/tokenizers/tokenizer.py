# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: tokenizer.py
@time: 2020/12/16 23:18:34

这一行开始写关于本文件的说明与解释


'''
from typing import List, Optional


class Tokenizer():
    def batch_tokenizer(self, texts: List[str]):
        return [self.tokenize(text) for text in texts]

    def tokenize(self, text):
        raise NotImplementedError

    def add_special_tokens(self, tokens1: List, tokens2: Optional[List]):
        """添加特殊tokens; 如 [CLS] or [SEP]等.

        Args:
            tokens1 (List): 原始tokens
            tokens2 (Optional[List]): 将要添加到tokens1中的tokens

        Returns:
        """
        return tokens1 + (tokens2 or [])
