# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: slot_gate_attention.py
@time: 2021/03/17 20:28:11

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf


class SlotGateAttention(tf.keras.layers.Layer):
    """
        get intent attention and slot attention
        reference: "Slot-Gated Modeling for Joint Slot Filling and Intent Prediction", Chih-Wen Goo, Guang Gao
        : remove_slot_attn: if False, use slot attention, else c_slot = state_outputs
    """

    def __init__(self, atten_size, remove_slot_attn=False, **kwargs):
        self.remove_slot_attn = remove_slot_attn
        super(SlotGateAttention, self).__init__(**kwargs)
        self.attn_size = atten_size
        self.intent_attention_v = self.add_weight(
            name="intent_AttnV", shape=(atten_size,),
            initializer="glorot_uniform")
        self.intent_input_dense = tf.keras.layers.Dense(
            self.attn_size, use_bias=True)
        self.intent_hidden_dense = tf.keras.layers.Dense(
            self.attn_size, use_bias=True)
        if not remove_slot_attn:
            self.slot_attention_v = self.add_weight(
                name="slot_AttnV", shape=(self.attn_size,),
                initializer="glorot_uniform"
            )
            self.slot_input_dense = tf.keras.layers.Dense(
                self.attn_size, use_bias=True)
            self.slot_hidden_dense = tf.keras.layers.Dense(
                self.attn_size, use_bias=True)

    def call(self, state_outputs, final_state, mask=None):
        # state_outputs: hidden 层输出
        # final_state: hidden 层最终状态
        slot_inputs = state_outputs  # h
        intent_input = final_state
        attn_size = state_outputs.shape[2]
        hidden_state = tf.expand_dims(state_outputs, axis=2)  # (b, s, 1, 2*e)
        # intent attention (BahdanauAttention,
        # Ref:NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE)
        intent_hidden_features = self.intent_hidden_dense(
            hidden_state)  # W1* h_{j}   (b, s, 1, 2*e)
        intent_input_features = self.intent_input_dense(
            intent_input)  # W2 * f  (b, 2*e)
        intent_input_features = tf.reshape(
            intent_input_features, [-1, 1, 1, attn_size])
        # (b, 1, 1, 2*e)
        # How Bahdanau compute the score, score = v * tanh(W1 * h_{j} + W2 * f)
        intent_score = tf.reduce_sum(
            self.intent_attention_v * tf.nn.tanh(intent_hidden_features + intent_input_features),
            axis=[2, 3])
        # obtain attention score, correlation, that is, the weight of the hj
        intent_weights = tf.nn.softmax(intent_score)
        intent_weights = tf.expand_dims(
            tf.expand_dims(intent_weights, -1), -1)
        # Weighted summation to get the intent context vector, \sum_{j} a_{i,j}*h_{j}
        c_intent = tf.reduce_sum(
            intent_weights * hidden_state, axis=[1, 2])  # (b, 2*e)
        c_slot = slot_inputs
        # slot attention (BahdanauAttention)
        if not self.remove_slot_attn:
            slot_hidden_features = self.slot_hidden_dense(
                hidden_state)  # W1* h_{j}
            origin_shape = tf.shape(state_outputs)
            slot_hidden_features = tf.reshape(
                slot_hidden_features, origin_shape)
            slot_hidden_features = tf.expand_dims(
                slot_hidden_features, axis=1)
            slot_input_feature = self.slot_input_dense(
                slot_inputs)  # W2 * s_{i}
            slot_input_feature = tf.expand_dims(
                slot_input_feature, axis=2)
            # Correlation between hj and si, score = v * tanh(W1* h_{j} + W2 * s_{i})
            slot_score = tf.reduce_sum(
                self.slot_attention_v * tf.nn.tanh(slot_hidden_features + slot_input_feature),
                axis=3)
            slot_weights = tf.nn.softmax(slot_score)  # attention score,相关度
            slot_weights = tf.expand_dims(slot_weights, axis=-1)
            slot_inputs = tf.expand_dims(slot_inputs, axis=1)
            # Weighted summation to get the slot context vector, \sum_{j} a_{i,j}*h_{j}
            c_slot = tf.reduce_sum(slot_weights * slot_inputs, axis=2)

        return c_slot, c_intent
