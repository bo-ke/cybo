# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: losses.py
@time: 2021/04/08 21:56:19

这一行开始写关于本文件的说明与解释


'''
from typing import TypeVar, Union
import tensorflow as tf


class Loss:

    def compute_losses(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError


Losses = TypeVar("Losses", bound=Union[Loss, tf.keras.losses.Loss])
