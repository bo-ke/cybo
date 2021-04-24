# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: transformers_pretrained_layer.py
@time: 2021/04/24 17:10:34

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from transformers import TFAutoModel, AutoConfig
from transformers.modeling_tf_bert import TFBertModel


class TransformersPretrainedLayer(tf.keras.layers.Layer):
    def __init__(
            self, pretrained_model: str, from_pt: bool = False, trainable=True,
            name=None, dtype=None, dynamic=False, **kwargs):
        super().__init__(trainable=trainable, name=name,
                         dtype=dtype, dynamic=dynamic, **kwargs)
        # _config = AutoConfig.from_pretrained(
        #     pretrained_model_name_or_path=pretrained_model)
        # self._bert = TFAutoModel(config=_config)
        self._bert = TFAutoModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model, from_pt=from_pt)

    def call(self, inputs, attention_mask=None, token_type_ids=None):
        return self._bert(
            inputs=inputs, attention_mask=attention_mask,
            token_type_ids=token_type_ids)
