# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: slu_loss.py
@time: 2021/03/04 00:35:01

这一行开始写关于本文件的说明与解释


'''
from typing import Dict
import tensorflow as tf


def slu_loss_func(y_true: Dict, y_pred: Dict):
    intent_true, slot_true = y_true
    intent_pred, slot_pred = y_pred

    mask = tf.cast(tf.math.not_equal(slot_true, 0), tf.float32)  # (b, s)
    mask = tf.expand_dims(mask, axis=-1)  # (b, s, 1)
    intent_true = tf.broadcast_to(
        intent_true, shape=intent_pred.shape[:-1])  # (b, s)
    intent_loss = tf.keras.losses.SparseCategoricalCrossentropy()(
        y_true=intent_true,
        y_pred=intent_pred*mask)
    slot_loss = tf.keras.losses.SparseCategoricalCrossentropy()(
        y_true=slot_true,
        y_pred=slot_pred*mask)
    return slot_loss + intent_loss
