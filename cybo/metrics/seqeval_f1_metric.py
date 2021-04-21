# -*- coding: utf-8 -*-
'''
@author: kebo
@contact: kebo0912@outlook.com

@version: 1.0
@file: seqeval_f1_metric.py
@time: 2021/04/18 20:25:27

这一行开始写关于本文件的说明与解释


'''
from typing import Dict, Tuple, List
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
import tensorflow as tf


class SeqevalF1Metric():
    def __init__(self) -> None:
        pass

    def _align_predictions(predictions: tf.Tensor, label_ids: tf.Tensor,
                           label_map: Dict) -> Tuple[List[int],
                                                     List[int]]:
        preds = tf.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != -100:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

            return preds_list, out_label_list

    def compute_metrics(self, predictions, label_ids) -> Dict:
        preds_list, out_label_list = self.align_predictions(
            predictions, label_ids)

        return {
            "precision": precision_score(out_label_list, preds_list),
            "recall": recall_score(out_label_list, preds_list),
            "f1": f1_score(out_label_list, preds_list),
        }
