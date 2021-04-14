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


class TransformersBertTokenizer(Tokenizer, BertTokenizer):
    """huggingface-transformers BertTokenizer

    Args:
        Tokenizer ([type]): [description]
    """
