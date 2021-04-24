# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: transformers_slu_dataset_reader.py
@time: 2021/04/25 00:03:59

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from typing import Optional, Dict, List

from cybo.data.vocabulary import Vocabulary
from cybo.data.dataset_readers.transformers_token_classification_dataset_reader import TransformersTokenClassificationDatasetReader, \
    TransformersTokenClassificationInputExample, TransformersTokenClassificationInputFeatures
from cybo.data.dataset_readers.slu_dataset_reader import SluInputExample, SluInputFeatures, SluDatasetReader


class TransformersSluInputExample(SluInputExample):
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for k, v in self.dict(exclude={"guid": ..., "text": ...}).items():
            if isinstance(v, list):
                for i in v:
                    counter[k][i] += 1
            else:
                counter[k][v] += 1


class TransformersSluInputFeatures(SluInputFeatures):
    attention_mask: List[int]
    token_type_ids: List[int]

    @classmethod
    def output_types(cls):
        return {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32,
                "intent_ids": tf.int32, "tags_ids": tf.int32}


class TransformersSluDatasetReader(
        TransformersTokenClassificationDatasetReader, SluDatasetReader):
    def _convert_example_to_features(
            self, example: TransformersSluInputExample, vocab: Vocabulary,
            max_seq_length: int) -> TransformersSluInputFeatures:
        _example = TransformersTokenClassificationInputExample(
            guid=example.guid, words=example.text, labels=example.tags)

        _features = super()._convert_example_to_features(
            example=_example, max_seq_length=max_seq_length,
            label_map=vocab._token_to_index["tags"])
        return TransformersSluInputFeatures(
            input_ids=_features.input_ids,
            attention_mask=_features.attention_mask,
            token_type_ids=_features.token_type_ids,
            intent_ids=[vocab._token_to_index["intent"].get(example.intent)],
            tags_ids=_features.label_ids)
