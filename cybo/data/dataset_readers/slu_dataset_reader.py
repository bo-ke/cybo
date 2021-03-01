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
from typing import List, Optional, Dict
import json
import re
import pandas as pd
from pydantic import BaseModel

from cybo.data.tokenizers.tokenizer import Tokenizer
from cybo.data.vocabulary import Vocabulary


class SluInputExample(BaseModel):
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


class InputFeatures(BaseModel):
    input_ids: List[int]
    intent_ids: Optional[List[int]]
    tags_ids: Optional[List[int]]

    @classmethod
    def return_types(cls):
        return {"input_ids": tf.int32, "intent_ids": tf.int32, "tags_ids": tf.int32}


class SluDatasetReader():
    def init(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def get_examples(self, data_filepath) -> List[SluInputExample]:
        def formatting_tags(slot_dict, query):
            tags = ["O"]*len(query)
            if not slot_dict:
                return tags
            for slot in slot_dict:
                pattern = re.match(r".\|(.*)", slot).group(1).upper()

                def update_tags(position_start, position_end, tags):
                    tags[position_start] = f"B_{pattern}"
                    for i in range(position_start+1, position_end+1):
                        tags[i] = f"I_{pattern}"

                if isinstance(slot_dict[slot], list):
                    for sub_slot in slot_dict[slot]:
                        update_tags(
                            sub_slot["position_start"],
                            sub_slot["position_end"],
                            tags)
                else:
                    update_tags(
                        slot_dict[slot]["position_start"],
                        slot_dict[slot]["position_end"], tags)

            return tags

        lines = self._read_file(filepath=data_filepath)
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text = self.tokenizer.tokenize(line["query"])
            intent = line["意图"][0]
            tags = formatting_tags(line.get("槽位", {}), line["query"])
            examples.append(SluInputExample(
                guid=guid, intent=intent, tags=tags, text=text))
        return examples

    @classmethod
    def _read_file(cls, filepath):
        df = pd.read_csv(filepath)
        lines = []
        for query, mark_result_json in zip(df["query"], df["markResultJson"]):
            mark_result = json.loads(mark_result_json)
            mark_result.update({"query": json.loads(query)})
            lines.append(mark_result)
        return lines

    def convert_examples_to_features(
            self, examples: List[SluInputExample],
            vocab: Vocabulary, max_seq_length: int = 32,
            return_generator: bool = False):
        features = []
        for (ex_index, example) in enumerate(examples):
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
            feature = InputFeatures(
                input_ids=input_ids, intent_ids=intent_ids,
                tags_ids=tags_ids)
            # if return_generator:
            #     print(return_generator)
            #     yield feature.dict(exclude_unset=True)
            features.append(feature)
        return features

    def encode_plus(self, text, *args, **kwargs) -> Dict:
        pass

    @classmethod
    def _truncated_add_padded(cls, tokens, max_seq_length, padding_token=0):
        if len(tokens) > max_seq_length:
            tokens = tokens[:max_seq_length]
        else:
            tokens = tokens + [padding_token] * (max_seq_length - len(tokens))
        return tokens

    @property
    def return_types(self):
        return InputFeatures.return_types()
