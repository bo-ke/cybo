# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: additive_attention.py
@time: 2021/05/13 13:13:45

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf


class AdditiveAttention(tf.keras.layers.Layer):
    """Bahdanau(2015) additive_attention/Bahdanau attention
    # TODO 使用tf.keras.layers.AdditiveAttention 替换
    """

    def __init__(self, attn_size, **kwargs):
        super().__init__(**kwargs)
        self.attn_size = attn_size
        self.input_dense = tf.keras.layers.Dense(attn_size)
        self.context_dense = tf.keras.layers.Dense(attn_size)
        self.attn_v = self.add_weight(
            name="attnV", shape=(attn_size,),
            initializer="glorot_uniform")

    def call(self, inputs, context, mask=None, training=True):
        """
        Args:
            inputs ([type]): [batch_size, TV, dim] - K/V
            context ([type]): [batch_size, Tq, dim] - Q
            mask ([type], optional): [description]. Defaults to None.
        1. Reshape `query` and `value` into shapes `[batch_size, Tq, 1, dim]`
            and `[batch_size, 1, Tv, dim]` respectively.
        2. Calculate scores with shape `[batch_size, Tq, Tv]` as a non-linear
            sum: `scores = tf.reduce_sum(tf.tanh(query + value), axis=-1)`
        3. Use scores to calculate a distribution with shape
            `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
        4. Use `distribution` to create a linear combination of `value` with
            shape `batch_size, Tq, dim]`:
            `return tf.matmul(distribution, value)`.
        """
        _input_hidden = self.input_dense(inputs)
        _context_hidden = self.context_dense(context)

        _input_hidden = tf.expand_dims(_input_hidden, axis=1)
        # [b, 1, Tv, dim]
        _context_hidden = tf.expand_dims(_context_hidden, axis=2)
        # [b, Tq, 1, dim]
        e = self.attn_v * tf.tanh(_input_hidden + _context_hidden)
        # Correlation between hj and si, score = v * tanh(W1* h_{j} + W2 * s_{i})
        # [b, Tq, Tv, dim]
        attention_weights = tf.reduce_sum(e, axis=-1)
        # [b, Tq, Tv]
        if mask is not None:
            attention_mask = tf.cast(
                (1 - tf.cast(mask, tf.float32)),
                tf.float32)  # (b, Tv)
            # attention_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
            attention_mask = attention_mask[:, tf.newaxis, :]  # (b, 1, Tv)
            attention_weights += (attention_mask * -1e9)
        attention_score = tf.nn.softmax(attention_weights)
        # [b, Tq, Tv]
        # return [b, Tq, dim]
        # Weighted summation to get the slot context vector, \sum_{j} a_{i,j}*h_{j}
        return tf.matmul(attention_score, inputs)
