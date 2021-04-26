# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: slot_gate.py
@time: 2021/03/17 20:25:32

这一行开始写关于本文件的说明与解释


'''
from typing import Dict
import tensorflow as tf

from cybo.models.model import Model
from cybo.data.vocabulary import Vocabulary
from cybo.modules.attentions import SlotGateAttention
# from cybo.metrics.slu_overall_acc_metric import SluOverallAcc, debug
from cybo.metrics.nlu_acc_metric import NluAccMetric
from cybo.metrics.seqeval_f1_metric import SeqEvalF1Metric
from cybo.losses.sequence_classification_loss import SequenceClassificationLoss
from cybo.losses.token_classification_loss import TokenClassificationLoss


class SlotGate(Model):

    def __init__(
            self, embedding_dim, hidden_dim, dropout_rate,
            vocab: Vocabulary = None, *args, **kwargs):

        super().__init__(vocab=vocab, *args, **kwargs)

        _vocab_size = self._vocab.get_vocab_size("text")
        _intent_size = self._vocab.get_vocab_size("intent")
        _slot_size = self._vocab.get_vocab_size("tags")

        self.embedding = tf.keras.layers.Embedding(
            _vocab_size, embedding_dim, mask_zero=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.slot_gate_attention = SlotGateAttention(
            attn_size=2*hidden_dim, remove_slot_attn=False)

        self.v = self.add_weight(
            name="v", shape=(2 * hidden_dim,),
            initializer="glorot_uniform")
        self.intent_liner_layer = tf.keras.layers.Dense(2*hidden_dim)

        self.intent_output_dense = tf.keras.layers.Dense(
            _intent_size, activation="softmax")
        self.slot_output_dense = tf.keras.layers.Dense(
            _slot_size, activation="softmax")
        self.intent_loss = SequenceClassificationLoss()
        self.slot_loss = TokenClassificationLoss()

    def init_metrics(self):
        return {"nlu_acc": NluAccMetric(), "f1_score": SeqEvalF1Metric(label_map=self._vocab._index_to_token["tags"])}

    def call(
            self, input_ids, intent_ids=None, tags_ids=None, mask=None,
            training=True) -> Dict:
        inputs = self.embedding(input_ids)  # b, s, e
        hidden, forward_h, forward_c, backword_h, backword_c = self.bi_lstm(
            inputs)  # (b, s, 2*e) (b, e) (b, e) (b, e) (b, e)
        hidden = self.dropout(hidden, training=training)
        final_state = tf.concat([forward_h, backword_h], axis=-1)
        # (b, 2*e)
        c_slot, c_intent = self.slot_gate_attention(hidden, final_state)
        # (b, s, 2*e) (b, 2*e)
        # formula(6) in paper: g = \sum(v * tanh(C_slot + W * C_intent))
        _c_intent = tf.expand_dims(c_intent, axis=1)
        _c_intent = tf.broadcast_to(_c_intent, c_slot.shape)
        _c_intent = self.intent_liner_layer(_c_intent)
        # (b, s, 2*e)
        g = self.v * tf.nn.tanh(c_slot + _c_intent)
        g = tf.reduce_sum(g, axis=-1)  # (b, s)
        g = tf.expand_dims(g, axis=-1)
        # formula(7) in paper: y^S_i = softmax(W_{hy}^S(h_i + c_i^S.g))
        y_slot = self.slot_output_dense(hidden+c_slot*g)
        y_intent = self.intent_output_dense(final_state + c_intent)

        output_dict = {"intent_logits": y_intent,
                       "slot_logits": y_slot}
        if intent_ids is not None and tags_ids is not None:
            _intent_loss = self.intent_loss.compute_loss(
                y_true=intent_ids, y_pred=y_intent)
            _slot_loss = self.slot_loss.compute_loss(
                y_true=tags_ids, y_pred=y_slot)
            output_dict["loss"] = _intent_loss + _slot_loss

            self._metrics["nlu_acc"].update_state(
                y_true=[intent_ids, tags_ids],
                y_pred=[y_intent, y_slot])
            if not training:
                self._metrics["f1_score"].update_state(y_true=tags_ids,
                                                       y_pred=y_slot)
        return output_dict
