# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: sequence_classification_loss.py
@time: 2021/04/16 00:32:41

这一行开始写关于本文件的说明与解释


'''
from typing import List
import tensorflow as tf
# from transformers.modeling_tf_bert import TFSequenceClassificationLoss

from cybo.losses.loss import Loss


class SequenceClassificationLoss(Loss):
    """Loss function suitable for sequence classification.
    refer: huggingface: /transformers/modeling_tf_utils/TFSequenceClassificationLoss
    """

    def __init__(self,
                 loss_fn: tf.keras.losses.
                 Loss = tf.keras.losses.SparseCategoricalCrossentropy(
                     from_logits=True)) -> None:
        super().__init__(loss_fn)
