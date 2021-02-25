# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: instance.py
@time: 2020/12/21 23:52:15

这一行开始写关于本文件的说明与解释


'''
from typing import Mapping, MutableMapping, Dict

from cybo.data.fields.field import Field
from cybo.data.vocabulary import Vocabulary


class Instance(Mapping[str, Field]):

    def __init__(self, fields: MutableMapping[str, Field]) -> None:
        self.fields = fields
        # indexed flag
        self.indexed = False

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """在instance级别，进行count_vocab操作，计算counter，用于生成vocab
            在vocabulary的from_instances方法中被调用

        Args:
            counter (Dict[str, Dict[str, int]]): vocab->counter
        """
        for field in self.fields.values():
            # 调用每一个field的count_vocab_items函数
            field.count_vocab_items(counter)

    def index_fields(self, vocab: Vocabulary) -> None:
        if not self.indexed:
            self.indexed = True
            for field in self.fields.values():
                field.index(vocab)
