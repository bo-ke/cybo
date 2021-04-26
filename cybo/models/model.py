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

from cybo.metrics.metric import Metric
from cybo.data.vocabulary import Vocabulary


class Model(tf.keras.models.Model):

    def __init__(self, vocab: Vocabulary = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._vocab = vocab
        self._metrics = self.init_metrics()

    def init_metrics(self) -> Dict[str, Metric]:
        return {}

    def call(self, inputs, training=True, mask=None) -> Dict:
        raise NotImplementedError

    def get_metrics(self, reset: bool = False, training=True) -> Dict[str, float]:
        metrics_to_return = {}
        for _, _metric in self._metrics.items():
            if not training or _metric.support_tf_function:
                metrics_to_return = {**metrics_to_return,
                                     **_metric.compute_metrics()}
        if reset:
            for _, _metric in self._metrics.items():
                _metric.reset_states()
        return metrics_to_return
