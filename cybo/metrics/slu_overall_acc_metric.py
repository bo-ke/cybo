# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: slu_acc_metric.py
@time: 2021/03/04 00:31:25

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf


class SluOverallAcc(tf.keras.metrics.Metric):
    def __init__(self, name="slu_overall_acc", dtype=None, **kwargs):
        super().__init__(name=name, dtype=dtype, **kwargs)
        self.init_shape = []

        def _zero_wt_init(name):
            return self.add_weight(
                name, shape=self.init_shape, initializer="zeros",
                dtype=self.dtype)
        self.intent_positive = _zero_wt_init("intent_positive")
        self.slot_positive = _zero_wt_init("slot_positive")
        self.positive = _zero_wt_init("positive")

        self.count = _zero_wt_init("count")

    def update_state(self, y_true, y_pred):
        intent_true, slot_true = y_true.values()
        intent_pred, slot_pred = y_pred.values()

        o_intent = tf.argmax(intent_pred, axis=-1)
        # 取token_level_intent most_common 作为query intent
        #  Counter.most_common() 返回一个list，.eg:[(2, 3)]
        mask = tf.cast(tf.math.not_equal(slot_true, 0), tf.int32)
        seq_length = tf.reduce_sum(mask, axis=-1)
        ###
        # https://www.tensorflow.org/api_docs/python/tf/unique_with_counts
        ###
        # o_intent = [[
        #     Counter(o_intent[i].numpy()[:seq_length[i]]).most_common(1)[0][0]]
        #     for i in range(len(o_intent))]

        def get_max_count_intent(_intent):
            _y, _idx, _count = tf.unique_with_counts(_intent)
            _intent = _y[tf.argmax(_count)]
            return [_intent]
        o_intent = tf.convert_to_tensor([
            get_max_count_intent(o_intent[i][: seq_length[i]]
                                 for i in range(len(seq_length)))])
        # tf.print(o_intent)
        o_slot = tf.cast(tf.argmax(slot_pred, axis=-1), dtype=tf.int32)

        intent_correct_prediction = tf.equal(
            tf.cast(o_intent, dtype=tf.int32),
            tf.cast(intent_true, dtype=tf.int32))
        intent_correct_prediction = tf.cast(
            intent_correct_prediction, dtype=tf.int32)

        # mask = tf.cast(tf.math.not_equal(slot_true, 0), tf.int32)
        slot_correct_prediction = tf.equal(
            tf.cast(o_slot*mask, dtype=tf.int32),
            tf.cast(slot_true, dtype=tf.int32))

        slot_correct_prediction = tf.reduce_mean(
            tf.cast(slot_correct_prediction, dtype=tf.int32), axis=-1)
        slot_correct_prediction = tf.expand_dims(
            slot_correct_prediction, axis=-1)
        prediction = intent_correct_prediction * slot_correct_prediction

        self.intent_positive.assign_add(
            tf.cast(tf.reduce_sum(intent_correct_prediction),
                    dtype=tf.float32))
        self.slot_positive.assign_add(
            tf.cast(tf.reduce_sum(slot_correct_prediction),
                    dtype=tf.float32))
        self.positive.assign_add(
            tf.cast(tf.reduce_sum(prediction),
                    dtype=tf.float32))
        self.count.assign_add(tf.cast(len(prediction), dtype=tf.float32))

    def result(self):
        return self.intent_positive/self.count, self.slot_positive/self.count, self.positive/self.count
