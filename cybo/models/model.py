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
    def call(self, inputs, training, mask):
        raise NotImplementedError

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {}

    def update_metrics_state(self, y_true, y_pred):
        raise NotImplementedError

    def get_loss(self, y_true, y_pred):
        raise NotImplementedError

    def get_output_dict(self, **kwargs) -> Dict[str, tf.Tensor]:
        """
        output1 = self(input1)
        output_dict = {"output1": output1}
        if targets is not None:
            # Function returning a scalar tf.Tensor, defined by the user.
            loss = self._compute_loss(output1, output2, targets)
            output_dict["loss"] = loss
        return output_dict
        """
        return self(**kwargs)
