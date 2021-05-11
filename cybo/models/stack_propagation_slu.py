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
from typing import Dict

from cybo.data.vocabulary import Vocabulary
from cybo.modules.attentions import SelfAttentionLayer
from cybo.losses.slu_loss import slu_loss_func
# from cybo.metrics.slu_overall_acc_metric import SluTokenLevelIntentOverallAcc
from cybo.metrics.nlu_acc_metric import NluAccMetric, Metric
from cybo.losses.token_classification_loss import TokenClassificationLoss
from cybo.models.model import Model


class StackPropagationSlu(Model):
    def __init__(
            self, embedding_dim, hidden_dim, dropout_rate,
            vocab: Vocabulary, *args, **kwargs):
        super().__init__(vocab=vocab, *args, **kwargs)

        _vocab_size = self._vocab.get_vocab_size("text")
        _intent_size = self._vocab.get_vocab_size("intent")
        _slot_size = self._vocab.get_vocab_size("tags")

        self.embedding = tf.keras.layers.Embedding(
            _vocab_size, embedding_dim, mask_zero=True)
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
            units=_intent_size)
        self.slot_liner_layer = tf.keras.layers.Dense(
            units=_slot_size)

        self.intent_embedding = tf.keras.layers.Embedding(_intent_size, 8)
        self.slot_embedding = tf.keras.layers.Embedding(_slot_size, 32)
        self._intent_loss = TokenClassificationLoss()
        self._slot_loss = TokenClassificationLoss()

    def init_metrics(self) -> Dict[str, Metric]:
        return {"nlu_acc": NluAccMetric()}

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
        # 注意不可用reshape  transpose与reshape结果是不一样的
        # 错误写法: tf.reshape(y_intent.stack(), [x.shape[0], x.shape[1], -1])
        y_intent = tf.transpose(y_intent.stack(), [1, 0, 2])
        y_slot = tf.transpose(y_slot.stack(), [1, 0, 2])

        o_intent = self.get_o_intent(intent_pred=y_intent, mask=x._keras_mask)

        output_dict = {"intent_logits": o_intent, "slot_logits": y_slot}
        if intent_ids is not None and tags_ids is not None:
            _intent_ids = tf.broadcast_to(intent_ids, tags_ids.shape)
            active_loss = tags_ids != -100

            _intent_loss = self._intent_loss.compute_loss(
                y_true=tf.boolean_mask(_intent_ids, active_loss),
                y_pred=tf.boolean_mask(y_intent, active_loss))
            _slot_loss = self._slot_loss.compute_loss(
                y_true=tags_ids, y_pred=y_slot)
            output_dict["loss"] = _intent_loss + _slot_loss
            self._metrics["nlu_acc"].update_state(
                y_true=[intent_ids, tags_ids],
                y_pred=[o_intent, y_slot])
        return output_dict

    @staticmethod
    def get_o_intent(intent_pred, mask):
        mask = tf.cast(mask, dtype=tf.int32)
        o_intent = tf.argmax(intent_pred, axis=-1)
        seq_lengths = tf.reduce_sum(mask, axis=-1)
        # 取token_level_intent most_common 作为query intent
        # https://www.tensorflow.org/api_docs/python/tf/unique_with_counts

        def get_max_count_intent(_intent):
            _y, _idx, _count = tf.unique_with_counts(_intent)
            _intent = _y[tf.argmax(_count)]
            return [_intent]

        o_intent = tf.convert_to_tensor(
            [get_max_count_intent(o_intent[i][: seq_lengths[i]])
             for i in range(len(seq_lengths))], dtype=tf.int32)
        return o_intent
