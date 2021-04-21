# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: nlu_acc_metric.py
@time: 2021/04/21 23:20:00

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf

from cybo.metrics.metric import Metric


class NluAccMetric(Metric):
    def __init__(
            self, name: str = "nlu_acc", suppord_tf_function: bool = True) -> None:
        super().__init__(name=name, suppord_tf_function=suppord_tf_function)

        self.positive = self._zero_variable_init()
        self.count = self._zero_variable_init()

    def update_state(self, y_true, y_pred):
        intent_true, slot_true = y_true
        intent_pred, slot_pred = y_pred

        intent_acc = tf.keras.metrics.sparse_categorical_accuracy(
            y_true=intent_true, y_pred=intent_pred)

        if len(slot_pred.shape) == 3:
            slot_pred = tf.argmax(slot_pred, axis=-1, output_type=tf.int32)
        mask = tf.cast(tf.math.not_equal(slot_true, -100), tf.int32)
        slot_acc = tf.equal(slot_pred * mask, slot_true * mask)

        slot_acc = tf.reduce_mean(tf.cast(slot_acc, dtype=tf.int32), axis=-1)
        nlu_acc = tf.cast(intent_acc, tf.int32) * slot_acc

        self.positive.assign_add(tf.cast(tf.reduce_sum(nlu_acc), tf.float32))
        self.count.assign_add(tf.constant(len(nlu_acc), dtype=tf.float32))

    def reset_states(self):
        self.positive.assign(0)
        self.count.assign(0)

    def compute_metrics(self):
        return {
            "nlu_acc": (self.positive/self.count).numpy()
        }
