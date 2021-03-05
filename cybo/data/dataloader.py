# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: dataloader.py
@time: 2021/02/23 00:59:40

这一行开始写关于本文件的说明与解释


'''
from typing import List, Dict
import tensorflow as tf
import numpy as np
from collections import defaultdict

from cybo.data.vocabulary import Vocabulary
from cybo.data.dataset_readers.dataset_reader import InputFeatures


class Dataloader():
    def __init__(self, dataset: tf.data.Dataset, dataset_size: int,
                 batch_size: int):
        self._dataset = dataset
        self._dataset_size = dataset_size
        self._batch_size = batch_size

    @classmethod
    def from_features(cls, features: List[InputFeatures], batch_size: int):
        features_dict_data = defaultdict(list)
        for feature in features:
            for namespace, ids in feature.dict(exclude_unset=True).items():
                features_dict_data[namespace].append(ids)
        features_slice_data = dict(features_dict_data)
        dataset = tf.data.Dataset.from_tensor_slices(features_slice_data)
        return cls(
            dataset=dataset, dataset_size=len(features),
            batch_size=batch_size)

    @classmethod
    def from_features_generator(
            cls, features_generator, generator_size: int, output_types: Dict,
            batch_size: int):
        dataset = tf.data.Dataset.from_generator(
            lambda: features_generator, output_types=(output_types))
        return cls(
            dataset=dataset, dataset_size=generator_size,
            batch_size=batch_size)

    def __iter__(self):
        dataset = self._dataset.batch(self._batch_size, drop_remainder=False)
        dataset = dataset.prefetch(1)
        for batch in dataset:
            yield batch

    def __len__(self):
        # return tf.data.experimental.cardinality(self._dataset)
        # 在dataset为generator时候，或tfRecord等情况 不支持，generator情况下返回 -2...
        # refer:  https://github.com/tensorflow/tensorflow/issues/26966
        return self._dataset_size

    @property
    def batch_size(self):
        return self._batch_size
