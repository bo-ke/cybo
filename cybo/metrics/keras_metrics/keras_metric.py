# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: keras_metric.py
@time: 2021/04/21 23:59:34

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf

from cybo.metrics.metric import Metric


class KerasMetric(tf.keras.metrics.Metric, Metric):
    def __init__(
            self, name, dtype=None, support_tf_function: bool = True, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.support_tf_function = support_tf_function

    def _zero_wt_init(self, name, init_shape=[], dtype=tf.int32):
        return self.add_weight(
            name=name, shape=init_shape, initializer="zeros", dtype=dtype
        )

    def reset_states(self):
        return super().reset_states()

    def update_state(self, y_true, y_pred):
        return super().update_state(y_true, y_pred)

    def compute_metrics(self):
        raise NotImplementedError
