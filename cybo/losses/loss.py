# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: losses.py
@time: 2021/04/08 21:56:19

这一行开始写关于本文件的说明与解释


'''
from typing import List
import tensorflow as tf


class Loss:
    def __init__(self, loss_fn: tf.keras.losses.Loss = None) -> None:
        self._loss_fn = loss_fn

    def compute_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor,
                     sample_weight=None) -> tf.Tensor:
        return self._loss_fn(y_true=y_true, y_pred=y_pred,
                             sample_weight=sample_weight)


def shape_list(x: tf.Tensor) -> List[int]:
    """
    refer: huggingface
    Deal with dynamic shape in tensorflow cleanly.

    Args:
        x (:obj:`tf.Tensor`): The tensor we want the shape of.

    Returns:
        :obj:`List[int]`: The shape of the tensor as a list.
    """
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]
