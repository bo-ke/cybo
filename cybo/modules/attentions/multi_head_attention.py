# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: multi_head_attention.py
@time: 2021/05/15 23:26:07

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, hidden_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        assert hidden_dim % num_heads == 0
        self.depth = hidden_dim // num_heads
        self.wq = tf.keras.layers.Dense(hidden_dim)
        self.wk = tf.keras.layers.Dense(hidden_dim)
        self.wv = tf.keras.layers.Dense(hidden_dim)

        self.dense = tf.keras.layers.Dense(hidden_dim)

    def split_heads(self, inputs, batch_size):
        _inputs = tf.reshape(
            inputs, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(_inputs, [0, 2, 1, 3])

    def scaled_product_dot_attention(self, q, k, v, mask=None):
        # mask: b, Tv
        # q -> [b, Tq, dim] k: [b, Tk, dim] v: [b, Tv, dim] Tk=Tv
        def create_padding_mask(mask):
            attention_mask = 1 - tf.cast(mask, tf.int32)
            return attention_mask[:, tf.newaxis, tf.newaxis, :]

        d_k = tf.cast(tf.shape(k)[-1], tf.float32)
        _score = tf.matmul(q, tf.transpose(k))
        # 缩放matmul_qk
        scaled_attention_logits = _score/tf.sqrt(d_k)
        # # b, num_heads, Tq, Tk
        if mask is not None:
            attention_mask = create_padding_mask(mask)
            tf.expand_dims(attention_mask, -1)
            scaled_attention_logits += attention_mask*-1e9

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # b, num_heads, Tq, depth
        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, inputs, training=True,  mask=None, **kwargs):
        batch_size = inputs.shape[0]
        q = self.wq(inputs)
        k = self.wk(inputs)
        v = self.wv(inputs)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_product_attention, attention_weights = self.scaled_product_dot_attention(
            q, k, v, mask)
        # b, num_heads, Tq, depth
        scaled_product_attention = tf.transpose(
            scaled_product_attention, [0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_product_attention, [
                                      batch_size, -1, self.hidden_dim])
        return self.dense(concat_attention), attention_weights
