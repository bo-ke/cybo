# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: stack_propagation_slu.py
@time: 2021/02/23 01:01:13

这一行开始写关于本文件的说明与解释


'''
from collections import Counter
import tensorflow as tf

from cybo.layers.self_attention import SelfAttentionLayer
from cybo.losses.slu_loss import slu_loss_func
from cybo.metrics.slu_overall_acc_metric import SluTokenLevelIntentOverallAcc


class StackPropagationSlu(tf.keras.models.Model):
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

        self.acc = SluTokenLevelIntentOverallAcc()

    @tf.function()
    def call(self, input_ids, intent_ids=None, tags_ids=None, mask=None,
             training=True):
        x = self.embedding(input_ids)    # (b, s, e)
        x = self.dropout1(x, training=training)
        h = self.bi_lstm(x)      # (b, s, 2e)
        c = self.attention_layer(h)   # (b, s, 2e)
        e = self.concat([h, c])
        h_intent = self.intent_decoder_layer(e)  # (b, s, e)
        y_intent = self.intent_output_layer(h_intent)  # (b, s, intent_size)

        h_slot = self.slot_decoder_layer(
            self.concat([y_intent, e]))  # (b, s, (intent_size+4*e))
        y_slot = self.slot_output_layer(h_slot)   # (b, s, slot_size)
        output_dict = {"intent_logits": y_intent, "slot_logits": y_slot}
        if intent_ids is not None and tags_ids is not None:
            y_true = [intent_ids, tags_ids]
            y_pred = [y_intent, y_slot]
            loss = slu_loss_func(y_true=y_true, y_pred=y_pred)
            output_dict["loss"] = loss
            self.acc.update_state(y_true=y_true, y_pred=y_pred)
            # get_intent_slot_positive(y_true=y_true, y_pred=y_pred)
        return output_dict

    def get_metrics(self, reset: bool = False):
        metrics_to_return = {"intent_acc": self.acc.result()[0].numpy(),
                             "slot_acc": self.acc.result()[1].numpy(),
                             "overall_acc": self.acc.result()[2].numpy()}
        if reset:
            self.acc.reset_states()
        return metrics_to_return

    @classmethod
    def extend_pre_config(cls):
        model_config = {
            "embedding_dim": 200
        }
        return cls(**model_config)
