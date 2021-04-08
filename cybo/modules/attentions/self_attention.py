# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: self_attention.py
@time: 2021/02/23 01:03:25

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf


class QKVAttention(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.wq = tf.keras.layers.Dense(units=hidden_dim)
        self.wk = tf.keras.layers.Dense(units=hidden_dim)
        self.wv = tf.keras.layers.Dense(units=output_dim)

        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, input_query, input_key, input_value, mask=None, **kwargs):
        q = self.wq(input_query)  # (b, len(q), e)
        k = self.wk(input_key)    # (b, len(k), e)
        v = self.wv(input_value)  # (b, len(v), e)

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (b, len(q), len(k))
        scaled_attention_logits = matmul_qk / \
            tf.sqrt(tf.cast(k.shape[-1], tf.float32))
        if mask is not None:
            attention_mask = tf.cast(
                (1 - tf.cast(mask, tf.float32)),
                tf.float32)  # (b, s)
            # attention_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
            attention_mask = attention_mask[:, tf.newaxis, :]  # (b, 1, s)
            scaled_attention_logits += (attention_mask * -1e9)
        attention_weights = tf.nn.softmax(
            scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
        attention = tf.matmul(attention_weights, v)
        return self.dropout(attention)


class SelfAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, output_dim, dropout_rate):
        super(SelfAttentionLayer, self).__init__()
        self.qkv_attention = QKVAttention(
            hidden_dim=hidden_dim, output_dim=output_dim,
            dropout_rate=dropout_rate)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.supports_masking = True

    def call(self, inputs, mask=None, **kwargs):
        attention = self.qkv_attention(inputs, inputs, inputs)
        return self.dropout(attention)
