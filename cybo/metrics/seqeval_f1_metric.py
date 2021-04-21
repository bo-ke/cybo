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

from cybo.metrics.metric import Metric

from cybo.common.logger import logger


class SeqEvalF1Metric(Metric):
    def __init__(self, label_map: Dict, name: str = "seqeval_f1_score",
                 suppord_tf_function: bool = False) -> None:
        super().__init__(name=name, suppord_tf_function=suppord_tf_function)

        self.label_map = label_map
        self.preds_list = [[]]
        self.out_label_list = [[]]

    def update_state(self, y_true, y_pred):

        _preds_list, _out_label_list = self._align_predictions(
            predictions=y_pred,
            label_ids=y_true, label_map=self.label_map)
        self.preds_list.extend(_preds_list)
        self.out_label_list.extend(_out_label_list)

    @staticmethod
    def _align_predictions(predictions: tf.Tensor, label_ids: tf.Tensor,
                           label_map: Dict) -> Tuple[List[int],
                                                     List[int]]:
        if len(predictions.shape) == 3:
            preds = tf.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape
        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != -100:
                    out_label_list[i].append(
                        label_map[label_ids[i][j].numpy()])
                    preds_list[i].append(label_map[preds[i][j].numpy()])
        return preds_list, out_label_list

    def compute_metrics(self) -> Dict:
        logger.info(
            f"\n{classification_report(self.out_label_list, self.preds_list)}")

        return {
            "precision": precision_score(self.out_label_list, self.preds_list),
            "recall": recall_score(self.out_label_list, self.preds_list),
            "f1": f1_score(self.out_label_list, self.preds_list),
        }

    def reset_states(self):
        self.out_label_list = [[]]
        self.preds_list = [[]]
