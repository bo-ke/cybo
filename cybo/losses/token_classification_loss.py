# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: token_classification_loss.py
@time: 2021/04/16 00:19:14

这一行开始写关于本文件的说明与解释


'''
from typing import List
import tensorflow as tf

from cybo.losses.loss import Loss, shape_list


class TokenClassificationLoss(Loss):
    """
    refer: huggingface: /transformers/modeling_tf_utils/TFTokenClassificationLoss

    Loss function suitable for token classification.

    .. note::

        Any label of -100 will be ignored (along with the corresponding logits) in the loss computation.

    """

    def __init__(
            self,
            loss_fn: tf.keras.losses.
            Loss = tf.keras.losses.SparseCategoricalCrossentropy()) -> None:
        super().__init__(loss_fn)

    def compute_loss(self, y_true, y_pred):
        # make sure only labels that are not equal to -100
        # are taken into account as loss
        active_loss = tf.reshape(y_true, (-1,)) != -100

        reduced_logits = tf.boolean_mask(tf.reshape(
            y_pred, (-1, shape_list(y_pred)[2])), active_loss)
        labels = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)

        return self._loss_fn(labels, reduced_logits)
