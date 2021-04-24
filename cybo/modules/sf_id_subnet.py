# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: sf_id_subnet.py
@time: 2021/04/24 15:12:20

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf


class SfIdSubnet(tf.keras.layers.Layer):
    """
        reference: "A Novel Bi-directional Interrelated Model for Joint Intent Detection and Slot Filling",
                  Haihong E , Peiqing Niu∗
        :priority_order: Either 'intent_first' or 'slot_first'.the former is ID_first mode, while the latter is
         SF_first mode
        :slot_n_tags: Positive integer, dimensionality of the output space
        :iteration_num: Positive integer, count of the Iteration Mechanism
        :id: which iteration
    """

    def __init__(self, attn_size, priority_order='slot_first',
                 iteration_num=1, id=0, **kwargs):
        super(SfIdSubnet, self).__init__(**kwargs)
        self.priority_order = priority_order
        self.iteration_num = iteration_num
        self.id = id

        self.attn_size = attn_size
        self.v = self.add_weight(
            name="AttnV", shape=(self.attn_size,),
            initializer="glorot_uniform")
        self.bias = self.add_weight(
            name="bias", shape=(self.attn_size,),
            initializer="glorot_uniform")
        self.gate_v = self.add_weight(
            name="gateV", shape=(self.attn_size,),
            initializer="glorot_uniform")
        self.intent_gate_dense = tf.keras.layers.Dense(self.attn_size)
        self.slot_dense = tf.keras.layers.Dense(self.attn_size)
        self.hidden_dense = tf.keras.layers.Dense(self.attn_size)

    def call(self, lstm_enc, final_state, c_slot, c_intent, mask=None):
        # slot_inputs = tf.reshape(lstm_enc, [-1, self.attn_size])
        slot_inputs = lstm_enc
        r_intent = c_intent
        intent_context_states = c_intent
        if self.priority_order == 'intent_first':
            slot_reinforce_state = tf.expand_dims(c_slot, axis=2)
            # intent subnet
            hidden = tf.expand_dims(lstm_enc, 2)
            slot_reinforce_features = self.slot_dense(
                slot_reinforce_state)  # v2 * c_slot
            hidden_features = self.hidden_dense(hidden)  # v1 * hi
            # formula (13) in paper : e = \sum(W * tanh(v1 * h_{i} + v2 * c_slot + b))
            e = tf.reduce_sum(
                self.v * tf.nn.tanh(hidden_features + slot_reinforce_features + self.bias), axis=[2, 3])
            # formula (12) in paper
            a = tf.nn.softmax(e)
            a = tf.expand_dims(tf.keras.expand_dims(a, axis=-1), axis=-1)
            # formula (11) in paper : \sum_{i}(a * c_slot)
            r = tf.reduce_sum(a * slot_reinforce_state, axis=[1, 2])
            r_intent = r + intent_context_states
            intent_output = tf.keras.layers.concatenate(
                [r_intent, final_state], 1)
            # slot subnet
            intent_gate = self.intent_gate_dense(r_intent)
            intent_gate = tf.reshape(
                intent_gate, [-1, 1, intent_gate.shape[1]])
            relation_factor = self.gate_v * tf.nn.tanh(c_slot + intent_gate)
            relation_factor = tf.reduce_sum(relation_factor, axis=2)
            relation_factor = tf.expand_dims(relation_factor, axis=-1)
            slot_reinforce_state1 = c_slot * relation_factor
            # slot_reinforce_vector = tf.reshape(
            #     slot_reinforce_state1, (-1, self.attn_size))
            slot_reinforce_vector = slot_reinforce_state1
            if self.id == self.iteration_num - 1:
                slot_output = tf.keras.layers.concatenate(
                    [slot_reinforce_vector, slot_inputs], -1)
                # slot_output = self.slot_n_tags_dense(slot_output)
                # slot_output = tf.reshape(
                #     slot_output, (-1, self.seq_len, self.attn_size))
                return [slot_output, intent_output, r_intent, slot_reinforce_state1]
            else:
                return [r_intent, slot_reinforce_state1]
        else:
            # slot subnet
            intent_gate = self.intent_gate_dense(r_intent)  # W * C_intent
            intent_gate = tf.reshape(
                intent_gate, [-1, 1, intent_gate.shape[1]])
            # formula(2) in paper: f = \sum(v * tanh(c_slot+ W* c_intent))
            relation_factor = self.gate_v * tf.nn.tanh(c_slot + intent_gate)
            relation_factor = tf.reduce_sum(relation_factor, axis=2)
            relation_factor = tf.expand_dims(relation_factor, axis=-1)
            # formula(3) in paper: r_slot = f * c_slot
            slot_reinforce_state = c_slot * relation_factor
            # slot_reinforce_vector = tf.reshape(
            #     slot_reinforce_state, (-1, self.attn_size))
            slot_reinforce_vector = slot_reinforce_state
            # intent subnet
            hidden = tf.expand_dims(lstm_enc, 2)
            slot_reinforce_output = tf.expand_dims(
                slot_reinforce_state, 2)
            slot_features = self.slot_dense(
                slot_reinforce_output)  # v1 * r_slot
            hidden_features = self.hidden_dense(hidden)  # v2 * hj
            # formula(6) in paper : e = \sum(W* tanh(v1 * r_slot + v2*hj + b))
            e = tf.reduce_sum(
                self.v * tf.nn.tanh(hidden_features + slot_features + self.bias),
                axis=[2, 3])
            # formula(5) in paper
            a = tf.nn.softmax(e)
            a = tf.expand_dims(tf.expand_dims(a, axis=-1), axis=-1)
            # formula(4) in paper: r = \sum_{i}(a * r_slot)
            r = tf.reduce_sum(a * slot_reinforce_output, axis=[1, 2])
            # formula(7) in paper : r_intent = r + c_intent
            r_intent = r + intent_context_states
            intent_output = tf.keras.layers.concatenate(
                [r_intent, final_state], 1)
            if self.id == self.iteration_num - 1:
                slot_output = tf.keras.layers.concatenate(
                    [slot_reinforce_vector, slot_inputs], -1)
                # slot_output = self.slot_n_tags_dense(slot_output)
                # slot_output = tf.reshape(
                #     slot_output, (-1, lstm_enc.shape[1], self.attn_size))
                return [slot_output, intent_output, r_intent, slot_reinforce_state]
            else:
                return [r_intent, slot_reinforce_state]
