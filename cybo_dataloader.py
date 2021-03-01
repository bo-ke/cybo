import tensorflow as tf
from typing import List, Optional, Dict
from pydantic import BaseModel
import pandas as pd
import json
from collections import defaultdict
import re
from cybo_vocabulary import Vocabulary


class IntentSlotInputExample(BaseModel):
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


class Tokenizer():
    def tokenize(self, text):
        text = text.strip()
        if not text:
            return []
        tokens = list(text)
        return tokens


# class Vocabulary():
#     def init(self) -> None:
#         self.token_to_index = {}
#         self.index_to_token = {}

#     @classmethod
#     def from_examples(
#             cls, examples: List[IntentSlotInputExample],
#             namespaces: List, non_padded_namespaces: List):
#         # for example in examples:
#         counter: Dict[str, Dict[str, int]] = defaultdict(
#             lambda: defaultdict(int))
#         for example in examples:
#             example.count_vocab_items(counter=counter)


class IntentSlotDatasetReader():
    def init(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def get_examples(self, data_filepath) -> List[IntentSlotInputExample]:
        def formatting_tags(slot_dict, query):
            tags = ["O"]len(query)
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
            examples.append(IntentSlotInputExample(
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
            self, examples: List[IntentSlotInputExample],
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


if name == "main":
    pass
