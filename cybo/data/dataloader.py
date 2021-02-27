# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: dataloader.py
@time: 2021/02/23 00:59:40

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
import numpy as np

from cybo.data.vocabulary import Vocabulary


class Dataloader():
    def __init__(self, instances, vocab: Vocabulary, batch_size,
                 max_seq_length):
        self.instances = instances
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.dataset = self._gen_dataset()

    def _data_generator(self):
        for instance in self.instances:
            text = self.vocab.encode(
                instance["text"],
                namespace="text", padding=True,
                padding_length=self.max_seq_length)
            mask = np.zeros(self.max_seq_length)
            mask[:len(instance["text"])] = 1
            intent = self.vocab.encode(
                [instance["intent"]] * len(instance["text"]),
                namespace="intent", padding=True,
                padding_length=self.max_seq_length)
            tag = self.vocab.encode(
                instance["tag"],
                namespace="tag", padding=True,
                padding_length=self.max_seq_length)
            # print({"text": text, "mask": mask, "intent": intent, "tag": tag})
            yield {"inputs": text, "mask": mask}, {"intent": intent, "tag": tag}

    def _gen_dataset(self):
        dataset = tf.data.Dataset.from_generator(self._data_generator, output_types=(
            {"inputs": tf.int32, "mask": tf.int32}, {"intent": tf.int32, "tag": tf.int32}))
        return dataset

    def __iter__(self):
        dataset = self.dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(1)
        for x, y in dataset:
            yield x, y

    def __len__(self):
        return len(self.instances)
