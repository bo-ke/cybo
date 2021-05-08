# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: crf_loss.py
@time: 2021/05/08 22:37:53

这一行开始写关于本文件的说明与解释


'''
from typing import List, Optional
import tensorflow as tf
from tensorflow_addons.text.crf import crf_log_likelihood, TensorLike

from cybo.losses.loss import Loss


class CrfLoss(Loss):
    """
    """

    def __init__(
            self,
            loss_fn=None) -> None:
        super().__init__(loss_fn)

    def compute_loss(self, inputs: TensorLike,
                     tag_indices: TensorLike,
                     sequence_lengths: TensorLike,
                     transition_params: Optional[TensorLike]) -> tf.Tensor:
        log_likelihood, transition_params = crf_log_likelihood(
            inputs, tag_indices, sequence_lengths, transition_params)
        return -tf.reduce_mean(log_likelihood)
