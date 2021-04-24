# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: bert_slu.py
@time: 2021/04/24 17:09:28

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from typing import Dict

from cybo.models.model import Model
from cybo.modules.transformers_pretrained_layer import TransformersPretrainedLayer
from cybo.losses.sequence_classification_loss import SequenceClassificationLoss
from cybo.losses.token_classification_loss import TokenClassificationLoss
from cybo.metrics.nlu_acc_metric import Metric, NluAccMetric


class BertSlu(Model):
    def __init__(self, pretrained_layer: TransformersPretrainedLayer,
                 dropout_rate, intent_size, slot_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_layer = pretrained_layer
        self.dropout1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.intent_output_dense = tf.keras.layers.Dense(
            intent_size, activation="softmax")
        self.slot_output_dense = tf.keras.layers.Dense(
            slot_size, activation="softmax")

        self.intent_loss = SequenceClassificationLoss()
        self.slot_loss = TokenClassificationLoss()

    def init_metrics(self) -> Dict[str, Metric]:
        return {"nlu_acc": NluAccMetric()}

    def call(
            self, input_ids, attention_mask, token_type_ids, intent_ids=None,
            tags_ids=None, training=True, mask=None) -> Dict:
        hidden_states, pooler = self.pretrained_layer(
            inputs=input_ids, attention_mask=attention_mask,
            token_type_ids=token_type_ids)
        pooler = self.dropout1(pooler, training=training)
        intent_logits = self.intent_output_dense(pooler)

        hidden_states = self.dropout2(hidden_states, training=training)
        slot_logits = self.slot_output_dense(hidden_states)

        output_dict = {"intent_logits": intent_logits,
                       "slot_logits": slot_logits}
        if intent_ids is not None and tags_ids is not None:
            _intent_loss = self.intent_loss.compute_loss(
                y_true=intent_ids, y_pred=intent_logits)
            _slot_loss = self.slot_loss.compute_loss(
                y_true=tags_ids, y_pred=slot_logits)
            output_dict["loss"] = _intent_loss + _slot_loss

            self._metrics["nlu_acc"].update_state(
                y_true=[intent_ids, tags_ids],
                y_pred=[intent_logits, slot_logits])
        return output_dict
