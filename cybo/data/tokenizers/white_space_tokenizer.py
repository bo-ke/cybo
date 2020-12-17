# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: white_space_tokenizer.py
@time: 2020/12/17 23:09:27

这一行开始写关于本文件的说明与解释


'''
from typing import List
from overrides import overrides

from cybo.data.tokenizers.tokenizer import Tokenizer
from cybo.data.tokenizers.token import Token


class WhiteSpaceTokenizer(Tokenizer):
    """
    空格分词
    """

    @overrides
    def tokenize(self, text: Lits[str]) -> List[Token]:
        return [Token(t) for t in text]
