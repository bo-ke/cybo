# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: slot_gated_attention.py
@time: 2021/03/17 20:28:11

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from cybo.modules.attentions.additive_attention import AdditiveAttention


class SlotGatedAttention(tf.keras.layers.Layer):
    """
        get intent attention and slot attention
        reference: "Slot-Gated Modeling for Joint Slot Filling and Intent Prediction", Chih-Wen Goo, Guang Gao
        : remove_slot_attn: if False, use slot attention, else c_slot = state_outputs
    """

    def __init__(self, attn_size, remove_slot_attn=False, **kwargs):
        self.remove_slot_attn = remove_slot_attn
        super(SlotGatedAttention, self).__init__(**kwargs)

        self.intent_attention = AdditiveAttention(attn_size=attn_size)
        if not remove_slot_attn:
            self.slot_attention = AdditiveAttention(attn_size=attn_size)

    def call(self, state_outputs, final_state, mask=None):
        # state_outputs: hidden 层输出
        # final_state: hidden 层最终状态
        slot_inputs = state_outputs  # h

        intent_input = tf.expand_dims(final_state, axis=1)
        _c_intent = self.intent_attention(
            inputs=state_outputs, context=intent_input)
        c_intent = tf.reduce_sum(_c_intent, axis=1)
        c_slot = slot_inputs
        # slot attention (BahdanauAttention)
        if not self.remove_slot_attn:
            c_slot = self.slot_attention(
                inputs=state_outputs, context=state_outputs)

        return c_slot, c_intent
