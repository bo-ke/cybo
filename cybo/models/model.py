# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: model.py
@time: 2021/02/23 00:52:46

这一行开始写关于本文件的说明与解释


'''
from typing import Dict
import tensorflow as tf


class Model(tf.keras.models.Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @tf.function()
    def call(self, inputs, training, mask) -> Dict:
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}
