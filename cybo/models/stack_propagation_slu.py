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

from cybo.modules.attentions import SelfAttentionLayer
from cybo.losses.slu_loss import slu_loss_func
from cybo.metrics.slu_overall_acc_metric import SluTokenLevelIntentOverallAcc
from cybo.models.model import Model


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

        self.intent_decoder_cell = tf.keras.layers.LSTMCell(units=64)
        self.slot_decoder_cell = tf.keras.layers.LSTMCell(units=64)
        self.intent_decoder_dropout = tf.keras.layers.Dropout(
            rate=dropout_rate)
        self.slot_decoder_dropout = tf.keras.layers.Dropout(rate=dropout_rate)

        self.intent_liner_layer = tf.keras.layers.Dense(
            units=intent_size, activation="softmax")
        self.slot_liner_layer = tf.keras.layers.Dense(
            units=slot_size, activation="softmax")

        self.intent_embedding = tf.keras.layers.Embedding(intent_size, 8)
        self.slot_embedding = tf.keras.layers.Embedding(slot_size, 32)

        self.acc = SluTokenLevelIntentOverallAcc()

    # @tf.function()
    def call(self, input_ids, intent_ids=None, tags_ids=None, mask=None,
             training=True):
        x = self.embedding(input_ids)    # (b, s, e)
        x = self.dropout1(x, training=training)
        h = self.bi_lstm(x)      # (b, s, 2e)
        c = self.attention_layer(h)   # (b, s, 2e)
        e = self.concat([h, c])

        # intent_decoder
        _intent_h_state, _intent_c_state = tf.zeros(
            [x.shape[0], 64]), tf.zeros([x.shape[0], 64])
        # (b, 64)
        _slot_h_state, _slot_c_state = tf.zeros(
            [x.shape[0], 64]), tf.zeros([x.shape[0], 64])
        # (b, 64)
        # https://stackoverflow.com/questions/64567161/tensorflow-cannot-be-accessed-here-it-is-defined-in-another-function-or-code-b
        y_intent, y_slot = tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True), tf.TensorArray(
            dtype=tf.float32, size=0, dynamic_size=True)
        # y_intent, y_slot = [], []
        prev_intent_tensor = tf.zeros([x.shape[0], 8])
        prev_slot_tensor = tf.zeros([x.shape[0], 32])
        for i in tf.range(x.shape[1]):
            _hidden = e[:, i, :]
            _intent_hidden = tf.concat([_hidden, prev_intent_tensor], axis=-1)
            # 添加dropout
            _intent_hidden = self.intent_decoder_dropout(
                _intent_hidden, training=training)
            _intent_h_state, (_intent_h_state, _intent_c_state) = self.intent_decoder_cell(
                _intent_hidden, states=[_intent_h_state, _intent_c_state])
            _h_intent_i = self.intent_liner_layer(_intent_h_state)
            y_intent = y_intent.write(i, _h_intent_i)
            # y_intent.append(_h_intent_i)
            prev_intent_tensor = self.intent_embedding(
                tf.argmax(_h_intent_i, axis=-1))
            # slot_decoder
            _slot_hidden = tf.concat(
                [_hidden, _h_intent_i, prev_slot_tensor],
                axis=-1)
            # 添加dropout
            _slot_hidden = self.slot_decoder_dropout(
                _slot_hidden, training=training)
            _slot_h_state, (_slot_h_state, _slot_c_state) = self.slot_decoder_cell(
                _slot_hidden, states=[_slot_h_state, _slot_c_state])
            _h_slot_i = self.slot_liner_layer(_slot_h_state)
            y_slot = y_slot.write(i, _h_slot_i)
            # y_slot.append(_h_slot_i)
            prev_slot_tensor = self.slot_embedding(
                tf.argmax(_h_slot_i, axis=-1))

        # y_intent = tf.stack(y_intent, axis=1)
        # y_slot = tf.stack(y_slot, axis=1)
        # 注意不可用reshape  transpose与reshape结果是不一样的
        # 错误写法: tf.reshape(y_intent.stack(), [x.shape[0], x.shape[1], -1])
        y_intent = tf.transpose(y_intent.stack(), [1, 0, 2])
        y_slot = tf.transpose(y_slot.stack(), [1, 0, 2])

        output_dict = {"intent_logits": y_intent, "slot_logits": y_slot}
        if intent_ids is not None and tags_ids is not None:
            y_true = {"intent": intent_ids, "tags": tags_ids}
            y_pred = {"intent_logits": y_intent, "slot_logits": y_slot}
            loss = slu_loss_func(y_true=y_true, y_pred=y_pred)
            output_dict["loss"] = loss
            self.acc.update_state(y_true=y_true, y_pred=y_pred)
            # debug(y_true=y_true, y_pred=y_pred)
        return output_dict

    def get_metrics(self, reset: bool = False):
        metrics_to_return = {"intent_acc": self.acc.result()[0].numpy(),
                             "slot_acc": self.acc.result()[1].numpy(),
                             "overall_acc": self.acc.result()[2].numpy()}
        if reset:
            self.acc.reset_states()
        return metrics_to_return
