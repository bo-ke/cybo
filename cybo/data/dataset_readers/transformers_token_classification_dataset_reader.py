# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: token_classification_dataset_reader.py
@time: 2021/04/15 00:42:38

这一行开始写关于本文件的说明与解释

'''
from typing import Optional, List, Dict
from cybo.data.dataset_readers.dataset_reader import DatasetReader, InputFeatures, InputExample
from cybo.data.tokenizers.transformers_bert_tokenizer import TransformersBertTokenizer


class TransformersTokenClassificationInputExample(InputExample):
    words: List[str]
    labels: Optional[List[str]]


class TransformersTokenClassificationInputFeatures(InputFeatures):
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class TransformersTokenClassificationDatasetReader(DatasetReader):
    def __init__(self, tokenizer: TransformersBertTokenizer) -> None:
        super().__init__(tokenizer=tokenizer)

    def get_examples(self, filepath) -> List[InputExample]:
        return super().get_examples(filepath)

    def _convert_example_to_features(
            self, example: InputExample,
            max_seq_length: int,
            label_map: Dict,
            cls_token_at_end=False,
            cls_token="[CLS]",
            cls_token_segment_id=1,
            sep_token="[SEP]",
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            pad_token_label_id=-100,
            sequence_a_segment_id=0,
            mask_padding_with_zero=True) -> InputFeatures:
        """from: https://github.com/huggingface/transformers/blob/master/examples/legacy/token-classification/utils_ner.py

        Args:
            example (InputExample): [description]
            vocab (Vocabulary): [description]
            max_seq_length (int): [description]
            cls_token_at_end (bool, optional): [description]. Defaults to False.
            cls_token (str, optional): [description]. Defaults to "[CLS]".
            cls_token_segment_id (int, optional): [description]. Defaults to 1.
            sep_token (str, optional): [description]. Defaults to "[SEP]".
            sep_token_extra (bool, optional): [description]. Defaults to False.
            pad_on_left (bool, optional): [description]. Defaults to False.
            pad_token (int, optional): [description]. Defaults to 0.
            pad_token_segment_id (int, optional): [description]. Defaults to 0.
            pad_token_label_id (int, optional): [description]. Defaults to -100.
            sequence_a_segment_id (int, optional): [description]. Defaults to 0.
            mask_padding_with_zero (bool, optional): [description]. Defaults to True.

        Returns:
            InputFeatures: [description]
        """
        # label_map = vocab._token_to_index["labels"]
        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = self._tokenizer.tokenize(word)

            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] +
                                 [pad_token_label_id] *
                                 (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = self._tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (
                    max_seq_length - special_tokens_count)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1]
                          * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id]
                           * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if "token_type_ids" not in self._tokenizer.model_input_names:
            segment_ids = None

        return TransformersTokenClassificationInputFeatures(
            input_ids=input_ids, attention_mask=input_mask,
            token_type_ids=segment_ids, label_ids=label_ids)
