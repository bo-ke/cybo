# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: slu_dataset_reader.py
@time: 2021/02/27 19:02:42

这一行开始写关于本文件的说明与解释


'''
import tensorflow as tf
from typing import List, Optional, Dict, Iterable
from tqdm import tqdm

from cybo.data.tokenizers.tokenizer import Tokenizer
from cybo.data.vocabulary import Vocabulary
from cybo.data.dataset_readers.dataset_reader import DatasetReader, InputExample, InputFeatures


class SluInputExample(InputExample):
    guid: int
    text: List[str]
    intent: str
    tags: List[str]

    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for token in self.text:
            counter["text"][token] += 1
        counter["intent"][self.intent] += 1
        for tag in self.tags:
            counter["tags"][tag] += 1


class SluInputFeatures(InputFeatures):
    input_ids: List[int]
    intent_ids: Optional[List[int]]
    tags_ids: Optional[List[int]]

    @classmethod
    def return_types(cls):
        return {"input_ids": tf.int32, "intent_ids": tf.int32, "tags_ids": tf.int32}


class SluDatasetReader(DatasetReader):
    def __init__(self, tokenizer: Tokenizer = None) -> None:
        super(SluDatasetReader, self).__init__(tokenizer=tokenizer)

    def get_examples(self, filepath) -> List[SluInputExample]:
        lines = self.read_file(filepath=filepath)
        examples = []
        for (i, line) in tqdm(enumerate(lines), desc=f"get examples from file: {filepath}"):
            examples.append(
                SluInputExample(
                    guid=i, intent=line["intent"][0],
                    tags=line["tags"],
                    text=line["text"]))
        return examples

    @classmethod
    def read_file(cls, filepath) -> Iterable[Dict]:
        lines = []
        text = []
        tags = []
        for line in open(filepath, "r", encoding="utf-8"):
            line = line.strip()
            if not line:
                text = []
                tags = []
                continue
            items = line.strip().split(" ", 1)
            if len(items) > 1:
                token, tag = items
                text.append(token)
                tags.append(tag)
            else:
                lines.append({"text": text, "intent": items, "tags": tags})
        return lines

    def _convert_example_to_features(
            self, example: SluInputExample, vocab: Vocabulary,
            max_seq_length: int = 32) -> InputFeatures:
        input_ids = [
            vocab.get_token_index(token, namespace="text")
            for token in example.text]
        intent_ids = [vocab.get_token_index(
            example.intent, namespace="intent")]
        tags_ids = [
            vocab.get_token_index(tag, namespace="tags")
            for tag in example.tags]
        input_ids = self._truncated_add_padded(
            input_ids, max_seq_length=max_seq_length)
        tags_ids = self._truncated_add_padded(
            tags_ids, max_seq_length=max_seq_length)
        return SluInputFeatures(
            input_ids=input_ids, intent_ids=intent_ids,
            tags_ids=tags_ids)
