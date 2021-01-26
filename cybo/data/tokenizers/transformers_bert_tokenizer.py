# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: bert_tokenizer.py
@time: 2021/01/26 23:37:11

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from transformers import BertTokenizer

from cybo.data.tokenizers import Tokenizer


class TransformersBertTokenizer(Tokenizer):
    """huggingface-transformers BertTokenizer

    Args:
        Tokenizer ([type]): [description]
    """

    def __init__(self) -> None:
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def encode_plus(self, text, max_length):
        tokenized = self.tokenizer.encode_plus(
            text, max_length=max_length, padding="max_length")
        input_ids = tf.convert_to_tensor(tokenized['input_ids'])
        attention_mask = tf.convert_to_tensor(tokenized['attention_mask'])
        token_type_ids = tf.convert_to_tensor(tokenized['token_type_ids'])
        return input_ids, attention_mask, token_type_ids
