# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: sf_id.py
@time: 2021/04/24 15:13:22

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from typing import Dict
import tensorflow_addons as tfa
from cybo.models.model import Model
from cybo.data.vocabulary import Vocabulary

from cybo.modules.slot_gated_attention import SlotGatedAttention
from cybo.modules.sf_id_subnet import SfIdSubnet
from cybo.losses.sequence_classification_loss import SequenceClassificationLoss
from cybo.losses.token_classification_loss import TokenClassificationLoss
from cybo.metrics.nlu_acc_metric import Metric, NluAccMetric
from cybo.metrics.seqeval_f1_metric import SeqEvalF1Metric
from cybo.losses.crf_loss import CrfLoss


class SfId(Model):
    def __init__(self, embedding_dim, hidden_dim, dropout_rate,
                 vocab: Vocabulary, priority_order: str = "slot_first",
                 iteration_num: int = 1, use_crf: bool = False, *args, **kwargs):
        super().__init__(vocab=vocab, *args, **kwargs)

        _vocab_size = self._vocab.get_vocab_size("text")
        _intent_size = self._vocab.get_vocab_size("intent")
        _slot_size = self._vocab.get_vocab_size("tags")

        self._use_crf = use_crf
        self.embedding = tf.keras.layers.Embedding(
            input_dim=_vocab_size, output_dim=embedding_dim, mask_zero=True)
        self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            hidden_dim, return_sequences=True, return_state=True))
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.slot_gated_attention = SlotGatedAttention(
            attn_size=2*hidden_dim, remove_slot_attn=False)

        self.sf_id_subnet_stack = [
            SfIdSubnet(
                attn_size=2 * hidden_dim,
                priority_order=priority_order, id=i,
                iteration_num=iteration_num)
            for i in range(iteration_num)]
        self.intent_output_dense = tf.keras.layers.Dense(
            _intent_size)
        self.slot_output_dense = tf.keras.layers.Dense(
            _slot_size)
        self.iteration_num = iteration_num
        self.intent_loss = SequenceClassificationLoss()
        if use_crf:
            self.crf = tfa.layers.CRF(_slot_size, use_kernel=False)
            self.slot_loss = CrfLoss()
        else:
            self.slot_loss = TokenClassificationLoss()

    def init_metrics(self) -> Dict[str, Metric]:
        return {"nlu_acc": NluAccMetric(), "f1_score": SeqEvalF1Metric(label_map=self._vocab._index_to_token["tags"])}

    def call(self, input_ids, intent_ids=None, tags_ids=None, mask=None,
             training=True) -> Dict:
        inputs = self.embedding(input_ids)
        hidden, forward_h, forward_c, backword_h, backword_c = self.bi_lstm(
            inputs)  # (b, s, 2*e) (b, e) (b, e) (b, e) (b, e)
        hidden = self.dropout(hidden, training=training)
        final_state = tf.concat([forward_h, backword_h], axis=-1)
        # (b, 2*e)
        c_slot, c_intent = self.slot_gated_attention(hidden, final_state)
        for _id, _sf_id_subnet in enumerate(self.sf_id_subnet_stack):
            if _id == self.iteration_num - 1:
                slot_output, intent_output, r_intent, slot_reinforce_state = _sf_id_subnet(
                    lstm_enc=hidden, final_state=final_state, c_slot=c_slot, c_intent=c_intent)
            else:
                r_intent, slot_reinforce_state = _sf_id_subnet(
                    lstm_enc=hidden, final_state=final_state, c_slot=c_slot, c_intent=c_intent)
                c_slot = slot_reinforce_state
                c_intent = r_intent
        y_slot = self.slot_output_dense(slot_output)
        y_intent = self.intent_output_dense(intent_output)
        output_dict = {"intent_logits": y_intent, "slot_logits": y_slot}
        if self._use_crf:
            decoded_sequence, potentials, sequence_length, chain_kernel = self.crf(
                y_slot)
            output_dict["decoded_sequence"] = decoded_sequence
        if intent_ids is not None and tags_ids is not None:
            _intent_loss = self.intent_loss.compute_loss(
                y_true=intent_ids, y_pred=y_intent)
            if self._use_crf:
                _slot_loss = self.slot_loss.compute_loss(
                    potentials, tags_ids, sequence_length, chain_kernel)
            else:
                _slot_loss = self.slot_loss.compute_loss(
                    y_true=tags_ids, y_pred=y_slot)
            output_dict["loss"] = _intent_loss + _slot_loss

            slot_pred = decoded_sequence if self._use_crf else y_slot
            self._metrics["nlu_acc"].update_state(
                y_true=[intent_ids, tags_ids],
                y_pred=[y_intent, slot_pred])
            if not training:
                self._metrics["f1_score"].update_state(y_true=tags_ids,
                                                       y_pred=slot_pred)
        return output_dict
