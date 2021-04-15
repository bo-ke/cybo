# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: dataloader.py
@time: 2021/02/23 00:59:40

这一行开始写关于本文件的说明与解释


'''
from typing import List
import tensorflow as tf
# import numpy as np
# from collections import defaultdict

from cybo.data.dataset_readers.dataset_reader import InputFeatures


class Dataloader():
    def __init__(self, dataset: tf.data.Dataset, dataset_size: int,
                 batch_size: int, shuffle: bool = True,
                 shuffle_buffer_size: int = None):
        self._dataset = dataset
        self._dataset_size = dataset_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        if shuffle:
            self._shuffle_buffer_size = shuffle_buffer_size or dataset_size

    @classmethod
    def from_features(
            cls, features: List[InputFeatures], batch_size: int):
        # features_dict_data = defaultdict(list)
        # for feature in features:
        #     for namespace, ids in feature.dict(exclude_unset=True).items():
        #         features_dict_data[namespace].append(ids)
        # features_slice_data = dict(features_dict_data)
        # dataset = tf.data.Dataset.from_tensor_slices(features_slice_data)
        def gen():
            for input_features in features:
                yield input_features.dict(exclude_unset=True)
        # 使用gen callable, 每个epoch都会生成一个新的generator
        output_types = features[0].output_types()
        dataset = tf.data.Dataset.from_generator(
            gen, output_types=output_types)
        return cls(
            dataset=dataset, dataset_size=len(features),
            batch_size=batch_size)

    def __iter__(self):
        dataset = self._dataset.batch(self._batch_size, drop_remainder=False)
        if self._shuffle:
            dataset.shuffle(buffer_size=self._shuffle_buffer_size)
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
