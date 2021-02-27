# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: stack_propagation_slu.py
@time: 2021/02/23 01:01:13

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from collections import Counter

from cybo.models.model import Model
from cybo.layers.self_attention import SelfAttentionLayer


class StackPropagationSlu(Model):

    def __init__(
            self, vocab_size, embedding_dim, hidden_dim, dropout_rate,
            intent_size, slot_size):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, mask_zero=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(hidden_dim, return_sequences=True))
        self.attention_layer = SelfAttentionLayer(
            hidden_dim=1024, output_dim=128,
            dropout_rate=dropout_rate)
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.concat = tf.keras.layers.Concatenate()

        self.intent_decoder_layer = tf.keras.layers.LSTM(
            units=64, return_sequences=True)
        self.slot_decoder_layer = tf.keras.layers.LSTM(
            units=64, return_sequences=True)

        self.intent_output_layer = tf.keras.layers.Dense(
            units=intent_size, activation="softmax")
        self.slot_output_layer = tf.keras.layers.Dense(
            units=slot_size, activation="softmax")

        # self.acc = IntentSlotOverallAcc()

    @tf.function()
    def call(self, inputs, mask=None, training=True):
        x = inputs
        x = self.embedding(x)    # (b, s, e)
        x = self.dropout1(x, training=training)
        h = self.bi_lstm(x)      # (b, s, 2*e)

        # attention_mask = tf.cast((1 - mask), tf.float32)  # (b, s)
        # # attention_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
        # attention_mask = attention_mask[:, tf.newaxis, :]  # (b, 1, s)
        c = self.attention_layer(h)   # (b, s, 2*e)
        # e = tf.concat(values=[h, c], axis=-1)  # (b, s, 3*e)
        e = self.concat([h, c])
        # print(c._keras_mask)
        # print(e._keras_mask)
        h_intent = self.intent_decoder_layer(e)  # (b, s, e)
        y_intent = self.intent_output_layer(h_intent)  # (b, s, intent_size)

        h_slot = self.slot_decoder_layer(
            self.concat([y_intent, e]))  # (b, s, (intent_size+4*e))
        y_slot = self.slot_output_layer(h_slot)   # (b, s, slot_size)
        # print(y_slot._keras_mask)
        return {"intent": y_intent, "tag": y_slot}

    @ classmethod
    def custom_predict(cls, *args, **kwargs):
        y_intent, y_slot = cls(*args, **kwargs)
        o_intent = tf.argmax(y_intent, axis=-1)
        # 取token_level_intent most_common 作为query intent
        #  Counter.most_common() 返回一个list，.eg:[(2, 3)]
        o_intent = [Counter(i.numpy()).most_common(1)[0][0] for i in o_intent]

        o_slot = tf.argmax(y_slot, axis=-1)
        return o_intent, o_slot

    def get_metrics(self, reset: bool = False):
        metrics_to_return = {"intent_acc": self.acc.result()[0].numpy(),
                             "slot_acc": self.acc.result()[1].numpy(),
                             "overall_acc": self.acc.result()[2].numpy()}
        if reset:
            self.acc.reset_states()
        return metrics_to_return

    def update_metrics_state(self, y_true, y_pred):
        self.acc.update_state(y_true=y_true, y_pred=y_pred)

    def get_loss(self, y_true, y_pred):
        # return intent_slot_loss_func(y_pred=y_pred, y_true=y_true)
        pass

    @classmethod
    def extend_pre_config(cls):
        model_config = {
            "embedding_dim": 200
        }
        return cls(**model_config)
